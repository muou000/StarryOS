//! vDSO data management.

extern crate axlog;
extern crate alloc;
use alloc::vec::Vec;

use axerrno::{AxError, AxResult};
use axhal::{
    mem::virt_to_phys,
    paging::MappingFlags,
};
use axmm::AddrSpace;
use kernel_elf_parser::{AuxEntry, AuxType};
use memory_addr::{MemoryAddr, PAGE_SIZE_4K};
use axalloc::{global_allocator, UsageKind};
use rand_pcg::rand_core::RngCore;
use rand_pcg::Pcg64Mcg;
use axhal::time::monotonic_time_nanos;

use starry_vdso::vdso::{VDSO_DATA, VdsoData, prepare_vdso_pages, vdso_data_paddr, VdsoAllocGuard};

/// Load vDSO into the given user address space and update auxv accordingly.
pub fn load_vdso_data(auxv: &mut Vec<AuxEntry>, uspace: &mut AddrSpace) -> AxResult<()> {
    let (vdso_start, vdso_end) = unsafe { starry_vdso::embed::init_vdso_symbols() };
    let (vdso_kstart, vdso_kend) = (vdso_start, vdso_end);

    const VDSO_USER_ADDR_BASE: usize = 0x7f00_0000;
    const VDSO_ASLR_PAGES: usize = 256;

    let seed: u128 = (monotonic_time_nanos() as u128)
        ^ ((vdso_kstart as u128).rotate_left(13))
        ^ ((vdso_kend as u128).rotate_left(37));
    let mut rng = Pcg64Mcg::new(seed);
    let page_off: usize = (rng.next_u64() as usize) % VDSO_ASLR_PAGES;
    let vdso_user_addr = VDSO_USER_ADDR_BASE + page_off * PAGE_SIZE_4K;
    axlog::info!("vdso_kstart: {vdso_kstart:#x}, vdso_kend: {vdso_kend:#x}",);

    if vdso_kend <= vdso_kstart {
        axlog::warn!("vDSO binary is missing or invalid: vdso_kstart={vdso_kstart:#x}, vdso_kend={vdso_kend:#x}. vDSO will not be loaded and AT_SYSINFO_EHDR will not be set.");
        return Err(AxError::InvalidExecutable);
    }

    let (vdso_paddr_page, vdso_bytes, vdso_size, vdso_page_offset, alloc_info) =
        prepare_vdso_pages(vdso_kstart, vdso_kend).map_err(|_| AxError::InvalidExecutable)?;

    let mut alloc_guard = VdsoAllocGuard::new(alloc_info);

    match kernel_elf_parser::ELFHeadersBuilder::new(vdso_bytes).and_then(|b| {
        let range = b.ph_range();
        b.build(&vdso_bytes[range.start as usize..range.end as usize])
    }) {
        Ok(headers) => {
            map_vdso_segments(headers, vdso_user_addr, vdso_paddr_page, vdso_page_offset, uspace)?;
            alloc_guard.disarm();
        }
        Err(_) => {
            // Fallback: map the whole vdso region as RX if parsing fails.
            axlog::info!("vDSO ELF parsing failed, using fallback mapping");
            uspace
                .map_linear(
                    vdso_user_addr.into(),
                    vdso_paddr_page + vdso_page_offset,
                    vdso_size,
                    MappingFlags::READ | MappingFlags::EXECUTE | MappingFlags::USER,
                )
                .map_err(|_| AxError::InvalidExecutable)?;
            alloc_guard.disarm();
        }
    }

    map_vvar_and_push_aux(auxv, vdso_user_addr, uspace)?;

    Ok(())
}

fn map_vvar_and_push_aux(auxv: &mut Vec<AuxEntry>, vdso_user_addr: usize, uspace: &mut AddrSpace) -> AxResult<()> {
    use starry_vdso::config::VVAR_PAGES;
    let vvar_user_addr = vdso_user_addr - VVAR_PAGES * PAGE_SIZE_4K;
    let mut alloc_vaddr_op: Option<usize> = None;
    let vvar_paddr = if VVAR_PAGES == 1 {
        vdso_data_paddr()
    } else {
        let num_pages = VVAR_PAGES;
        let alloc_vaddr = match global_allocator().alloc_pages(num_pages, PAGE_SIZE_4K, UsageKind::Global) {
            Ok(a) => a,
            Err(_) => return Err(AxError::InvalidExecutable),
        };
        alloc_vaddr_op = Some(alloc_vaddr);
        let dest = alloc_vaddr as *mut u8;
        let src = core::ptr::addr_of!(VDSO_DATA) as *const u8;
        let copy_len = core::mem::size_of::<VdsoData>();
        unsafe {
            core::ptr::copy_nonoverlapping(src, dest, copy_len);
            // Zero the rest of the allocated VVAR pages
            if num_pages * PAGE_SIZE_4K > copy_len {
                core::ptr::write_bytes(dest.add(copy_len), 0u8, num_pages * PAGE_SIZE_4K - copy_len);
            }
        }
        virt_to_phys(alloc_vaddr.into()).into()
    };

    if VVAR_PAGES > 1 {
        if uspace
            .map_linear(
                vvar_user_addr.into(),
                vvar_paddr.into(),
                VVAR_PAGES * PAGE_SIZE_4K,
                MappingFlags::READ | MappingFlags::USER,
            ).is_err()
        {
            if let Some(a) = alloc_vaddr_op {
                global_allocator().dealloc_pages(a, VVAR_PAGES, UsageKind::Global);
            }
            return Err(AxError::InvalidExecutable);
        }
    } else {
        uspace
            .map_linear(
                vvar_user_addr.into(),
                vvar_paddr.into(),
                VVAR_PAGES * PAGE_SIZE_4K,
                MappingFlags::READ | MappingFlags::USER,
            )
            .map_err(|_| AxError::InvalidExecutable)?;
    }

    axlog::info!(
        "Mapped vvar pages at user {:#x}..{:#x} -> paddr {:#x}",
        vvar_user_addr,
        vvar_user_addr + VVAR_PAGES * PAGE_SIZE_4K,
        vvar_paddr,
    );

    let aux_entry = AuxEntry::new(AuxType::SYSINFO_EHDR, vdso_user_addr);
    auxv.push(aux_entry);

    Ok(())
}

fn map_vdso_segments(
    headers: kernel_elf_parser::ELFHeaders,
    vdso_user_addr: usize,
    vdso_paddr_page: axhal::mem::PhysAddr,
    vdso_page_offset: usize,
    uspace: &mut AddrSpace,
) -> AxResult<()> {
    axlog::info!("vDSO ELF parsed successfully, mapping segments");
    for ph in headers.ph.iter().filter(|ph| ph.get_type() == Ok(xmas_elf::program::Type::Load)) {
        let vaddr = ph.virtual_addr as usize;
        let seg_pad = vaddr.align_offset_4k();
        let seg_align_size = (ph.mem_size as usize + seg_pad + PAGE_SIZE_4K - 1) & !(PAGE_SIZE_4K - 1);
        let seg_user_start = vdso_user_addr + vaddr.align_down_4k();
        let seg_paddr = vdso_paddr_page + vdso_page_offset + vaddr.align_down_4k();

        let mapping_flags = |flags: xmas_elf::program::Flags| -> MappingFlags {
            let mut mapping_flags = MappingFlags::USER;
            if flags.is_read() {
                mapping_flags |= MappingFlags::READ;
            }
            if flags.is_write() {
                mapping_flags |= MappingFlags::WRITE;
            }
            if flags.is_execute() {
                mapping_flags |= MappingFlags::EXECUTE;
            }
            mapping_flags
        };

        let flags = mapping_flags(ph.flags);
        uspace
            .map_linear(seg_user_start.into(), seg_paddr, seg_align_size, flags)
            .map_err(|_| AxError::InvalidExecutable)?;
    }
    Ok(())
}
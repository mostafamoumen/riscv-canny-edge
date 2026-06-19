#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

// RISC-V Linux ABI System Call Numbers
#define SYS_openat 56
#define SYS_close  57
#define SYS_lseek  62
#define SYS_read   63
#define SYS_write  64
#define SYS_fstat  80
#define SYS_exit   93
#define SYS_brk    214

#define AT_FDCWD -100

extern "C" {

// Inline assembly to trigger a RISC-V environment call (ecall)
static inline long syscall(long num, long a0=0, long a1=0, long a2=0, long a3=0) {
    register long a0_asm __asm__("a0") = a0;
    register long a1_asm __asm__("a1") = a1;
    register long a2_asm __asm__("a2") = a2;
    register long a3_asm __asm__("a3") = a3;
    register long a7_asm __asm__("a7") = num;
    __asm__ volatile ("ecall"
                      : "+r"(a0_asm)
                      : "r"(a1_asm), "r"(a2_asm), "r"(a3_asm), "r"(a7_asm)
                      : "memory");
    return a0_asm;
}

// ---------------------------------------------------------
// Newlib C Library Stubs (Used by std::ifstream / std::ofstream)
// ---------------------------------------------------------

int _open(const char *name, int flags, int mode) {
    int linux_flags = 0;
    
    // 1. Translate Basic Read/Write Access Flags
    if (flags & 1)        linux_flags |= 1; // O_WRONLY
    if (flags & 2)        linux_flags |= 2; // O_RDWR
    
    // 2. Translate Creation/Behavior Flags (Newlib -> Linux Kernel)
    if (flags & 0x0008)   linux_flags |= 0x400; // O_APPEND
    if (flags & 0x0200)   linux_flags |= 0x40;  // O_CREAT
    if (flags & 0x0400)   linux_flags |= 0x200; // O_TRUNC
    if (flags & 0x0800)   linux_flags |= 0x80;  // O_EXCL

    // 3. Give it default read/write permissions if creating a new file
    if ((linux_flags & 0x40) && mode == 0) {
        mode = 0666; 
    }

    // RISC-V uses openat instead of open
    return syscall(SYS_openat, AT_FDCWD, (long)name, linux_flags, mode);
}

int _close(int file) {
    return syscall(SYS_close, file);
}

int _read(int file, char *ptr, int len) {
    return syscall(SYS_read, file, (long)ptr, len);
}

int _write(int file, char *ptr, int len) {
    return syscall(SYS_write, file, (long)ptr, len);
}

int _lseek(int file, int ptr, int dir) {
    return syscall(SYS_lseek, file, ptr, dir);
}

int _fstat(int file, struct stat *st) {
    st->st_mode = 8192;
    return 0;
}

int _isatty(int file) {
    // Standard input, output, and error are ttys.
    return (file == 0 || file == 1 || file == 2) ? 1 : 0;
}

// Required for memory allocation (new/malloc) to grow the heap
void *_sbrk(int incr) {
    static long heap_end = 0;
    if (heap_end == 0) {
        heap_end = syscall(SYS_brk, 0); // Get current break
    }
    long prev_heap_end = heap_end;
    heap_end = syscall(SYS_brk, prev_heap_end + incr);
    return (void *)prev_heap_end;
}

void _exit(int status) {
    syscall(SYS_exit, status);
    while (1); // Halt
}

// Dummy stubs to prevent linker errors
int _kill(int pid, int sig) { (void)pid; (void)sig; return -1; }
int _getpid() { return 1; }

} // extern "C"
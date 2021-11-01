#ifndef MEMMAN_HEADER
#define MEMMAN_HEADER

#include <vector>
#include <string.h>

#undef STBI_MALLOC
#define STBI_MALLOC(sz) MemoryManager::MMmalloc(sz)
#define STBI_REALLOC(p,newsz)     MemoryManager::MMrealloc(p,newsz)
#define STBI_FREE(p)              MemoryManager::MMfree(p)
#undef STBI_REALLOC_SIZED
#define STBI_REALLOC_SIZED(p,oldsz,newsz) MemoryManager::MMrealloc_sized(p,oldsz,newsz)

void* malloc_werr(size_t sz); 
void* realloc_werr(void* p, size_t sz);

struct mem {
    void* ptr;
    bool is_free;
    size_t size;
    bool is_pinned;
    mem(size_t sz) {
        ptr = malloc_werr(sz);
        is_free = false;
        size = sz;
        is_pinned = false;
    }
};

class MemoryManager {
    static std::vector<mem> mems;
    static std::vector<mem> pow2s;

    static bool ispow2(size_t sz) { return (sz & (sz - 1)) == 0; }
    static size_t round2pos(size_t sz) { size_t f = 1;  while (f < sz) { f <<= 1; } return f; }
    static void* MMmallocpow2(size_t sz);
    static void* MMreallocpow2(void* p, size_t new_sz);
    static bool MMfreepow2(void* p);
public:
    static void* MMmalloc(size_t sz);
    static void* MMrealloc(void* p, size_t new_sz);
    static void* MMrealloc_sized(void* p, size_t old_sz, size_t new_sz);
    static void MMfree(void* p);
    static void MMpin(void* p, size_t size, unsigned flags = 0);
    ~MemoryManager();
};

#endif //MEMMAN_HEADER
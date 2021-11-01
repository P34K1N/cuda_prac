#include "memman.cuh"
#include "safecalls.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>


std::vector<mem> empty;
std::vector<mem> MemoryManager::mems(empty);
std::vector<mem> MemoryManager::pow2s(empty);


void* malloc_werr(size_t sz) {
    void* ptr = malloc(sz);
    if (ptr == NULL) {
        printf("Cannot allocate memory\n");
        throw 1;
    }
    return ptr;
}


void* realloc_werr(void *p, size_t sz) {
    void* ptr = realloc(p, sz);
    if (ptr == NULL) {
        printf("Cannot allocate memory\n");
        throw 1;
    }
    return ptr;
}


void* MemoryManager::MMmallocpow2(size_t sz) {
    for (int i = 0; i < pow2s.size(); i++) {
        if (pow2s[i].is_free) {
            pow2s[i].is_free = false;
            //printf("Malloc existing pow2 memory: %zu %p\n", sz, x.ptr);
            return pow2s[i].ptr;
        }
    }
    mem new_entry(sz);
    pow2s.push_back(new_entry);
    //printf("Malloc new pow2 memory: %zu %p\n", sz, new_entry.ptr);
    return new_entry.ptr;
}


void* MemoryManager::MMmalloc(size_t sz) {
    if (ispow2(sz)) {
        return MMmallocpow2(sz);
    }

    mem* best_guess = NULL;
    for (int i = 0; i < mems.size(); i++) {
        if (mems[i].is_free && mems[i].size >= sz && (
            best_guess == NULL || best_guess->size > mems[i].size)) {
            best_guess = &mems[i];
        }
    }
    if (best_guess != NULL) {
        best_guess->is_free = false;
        //printf("Malloc existing memory: %zu %p\n", sz, best_guess->ptr);
        return best_guess->ptr;
    }
    mem new_entry(sz);
    mems.push_back(new_entry);
    //printf("Malloc new memory: %zu %p\n", sz, new_entry.ptr);
    return new_entry.ptr;
}


void* MemoryManager::MMrealloc(void * p, size_t new_sz) {
    for (int i = 0; i < mems.size(); i++) {
        if (mems[i].ptr == p) {
            mems[i].ptr = realloc_werr(p, new_sz);
            mems[i].size = new_sz;
            //printf("Realloc: %zu %p\n", new_sz, x.ptr);
            return mems[i].ptr;
        }
    }
    mem new_entry(new_sz);
    mems.push_back(new_entry);
    //printf("Realloc to new memory: %zu, %p to %p\n", new_sz, p, new_entry.ptr);
    return new_entry.ptr;
}


void* MemoryManager::MMreallocpow2(void* p, size_t new_sz) {
    for (int i = 0; i < pow2s.size(); i++) {
        if (pow2s[i].ptr == p) {
            if (pow2s[i].size < new_sz) {
                size_t rs = round2pos(new_sz);
                pow2s[i].ptr = realloc_werr(p, rs);
                pow2s[i].size = rs;
                //printf("True realloc pow2 memory: %zu, %p to %p\n", new_sz, p, x.ptr);
                p = pow2s[i].ptr;
            }
            else {
                //printf("False realloc pow2 memory: %zu, %p\n", new_sz, p);
            }
            
            return p;
        }
    }
    return NULL;
}


void* MemoryManager::MMrealloc_sized(void* p, size_t old_sz, size_t new_sz) {
    if (old_sz == 0) {
        return MMmalloc(new_sz);
    }

    if (ispow2(old_sz)) {
        void* res = MMreallocpow2(p, new_sz);
        if (res != NULL) return res;
    }

    for (int i = 0; i < mems.size(); i++) {
        if (mems[i].ptr == p) {
            mems[i].is_free = true;
        }
    }
    
    mem* best_guess = NULL;
    for (int i = 0; i < mems.size(); i++) {
        if (mems[i].is_free && mems[i].size >= new_sz && (
            best_guess == NULL || best_guess->size > mems[i].size)) {
            best_guess = &mems[i];
        }
    }
    if (best_guess != NULL) {
        best_guess->is_free = false;
        memcpy(best_guess->ptr, p, old_sz);
        //printf("Realloc to existing memory: %zu to %zu, %p to %p\n", old_sz, new_sz, p, best_guess->ptr);
        return best_guess->ptr;
    }
    mem new_entry(new_sz);
    mems.push_back(new_entry);
    memcpy(new_entry.ptr, p, old_sz);
    //printf("Realloc to new memory: %zu to %zu, %p to %p\n", old_sz, new_sz, p, new_entry.ptr);
    return new_entry.ptr;
}


bool MemoryManager::MMfreepow2(void* p) {
    for (int i = 0; i < pow2s.size(); i++) {
        if (pow2s[i].ptr == p) {
            pow2s[i].is_free = true;
            //printf("Pow2 Free: %p\n", x.ptr);
            return true;
        }
    }
    return false;
}


void MemoryManager::MMfree(void* p) {
    if (MMfreepow2(p)) return;

    for (int i = 0; i < mems.size(); i++) {
        if (mems[i].ptr == p) {
            mems[i].is_free = true;
            //printf("Free: %p\n", x.ptr);
        }
    }
}


void MemoryManager::MMpin(void* p, size_t sz, unsigned flags) {
    if (ispow2(sz)) {
        for (int i = 0; i < pow2s.size(); i++) {
            if (pow2s[i].ptr == p) {
                if (!pow2s[i].is_pinned) {
                    SAFE_CALL(cudaHostRegister(p, sz, flags));
                    pow2s[i].is_pinned = true;
                }
                return;
            }
        }
    }

    for (int i = 0; i < mems.size(); i++) {
        if (mems[i].ptr == p) {
            if (!mems[i].is_pinned) {
                SAFE_CALL(cudaHostRegister(p, sz, flags));
                mems[i].is_pinned = true;
            }
            return;
        }
    }
}


MemoryManager::~MemoryManager() {
    for (int i = 0; i < mems.size(); i++) {
        free(mems[i].ptr);
    }
    for (int i = 0; i < pow2s.size(); i++) {
        free(pow2s[i].ptr);
    }
}

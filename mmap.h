#pragma once

#include <stdio.h>

#ifdef _WIN32
    #include <windows.h>
    #include <io.h>
    
    inline void*  memory_map(const char* path, size_t* size)
    {
        HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        LARGE_INTEGER fileSize;
        GetFileSizeEx(hFile, &fileSize);
        *size = (size_t)fileSize.QuadPart;
        HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        void* data = MapViewOfFile(hMap, FILE_MAP_READ,0,0,0);
        CloseHandle(hFile);
        return data;
    }
    inline void memory_unmap(void* ptr, size_t size)
    {
        UnmapViewOfFile(ptr);
    }

#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>

    inline void* memory_map(const char* path, size_t* size)
    {
        int fd = open(path, O_RDONLY);
        if (fd == -1) {
            if (size) *size = 0;
            return nullptr;
        }

        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 0) {
            close(fd);
            if (size) *size = 0;
            return nullptr;
        }

        *size = (size_t)st.st_size;
        if (*size == 0) {
            close(fd);
            return nullptr;
        }

        void* data = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (data == MAP_FAILED) {
            if (size) *size = 0;
            return nullptr;
        }

        return data;
    }

    inline void memory_unmap(void* ptr, size_t size)
    {
        if (ptr && size > 0) {
            munmap(ptr, size);
        }
    }
#endif


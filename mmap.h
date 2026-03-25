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
    #include <fcntl.h>

    inline void* memory_map(const char* path, size_t* size)
    {
        int fd = open(path, ORDONLY);
        struct stat st;
        fstat(fd, &st);
        *size = st.st_size;
        void* data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE,fd,0);
        close(fd);
        return data;
    }

    inline void memory_unmap(void* ptr, size_t size)
    {
        munmap(ptr,size);
    }
#endif


# 指定编译器和编译选项
CC = gcc
CFLAGS = -g
LDFLAGS =

# 源文件目录，存放有关CXL改动的源文件，如test.c、ssd.c等
SRCDIR = src
# 头文件目录，存放有关CXL改动的头文件，如test.h、ssd.h等
INCDIR = include
# 目标文件目录，存放编译生成的目标文件，如test.o、ssd.o等
BUILDDIR = build
# 第三方库目录，存放第三方库的源文件，如avlTree、cJSON等
LIBDIR = lib

# 确保生成目录存在
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# 默认目标
all: $(BUILDDIR) main

# 清理生成文件
clean:
	rm -rf $(BUILDDIR) main *.o *~
.PHONY: clean

# 链接目标文件生成可执行文件
main: $(BUILDDIR)/avlTree.o $(BUILDDIR)/flash.o $(BUILDDIR)/initialize.o $(BUILDDIR)/pagemap.o $(BUILDDIR)/readJson.o $(BUILDDIR)/cJSON.o $(BUILDDIR)/ssd.o $(BUILDDIR)/main.o
	$(CC) $(LDFLAGS) -o main $(BUILDDIR)/avlTree.o $(BUILDDIR)/flash.o $(BUILDDIR)/initialize.o $(BUILDDIR)/pagemap.o $(BUILDDIR)/readJson.o $(BUILDDIR)/cJSON.o $(BUILDDIR)/ssd.o $(BUILDDIR)/main.o

# 编译源文件生成目标文件
$(BUILDDIR)/avlTree.o: $(LIBDIR)/avlTree/avlTree.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $(LIBDIR)/avlTree/avlTree.c -o $@

$(BUILDDIR)/cJSON.o: $(LIBDIR)/cJSON/cJSON.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $(LIBDIR)/cJSON/cJSON.c -o $@

$(BUILDDIR)/flash.o: $(INCDIR)/pagemap.h $(SRCDIR)/flash.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/flash.c -o $@

$(BUILDDIR)/initialize.o: $(LIBDIR)/avlTree/avlTree.h $(INCDIR)/pagemap.h $(SRCDIR)/initialize.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/initialize.c -o $@

$(BUILDDIR)/pagemap.o: $(INCDIR)/initialize.h $(SRCDIR)/pagemap.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/pagemap.c -o $@

$(BUILDDIR)/readJson.o: $(SRCDIR)/readJson.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/readJson.c -o $@

$(BUILDDIR)/ssd.o: $(INCDIR)/flash.h $(INCDIR)/initialize.h $(INCDIR)/pagemap.h $(SRCDIR)/ssd.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/ssd.c -o $@

$(BUILDDIR)/main.o: $(SRCDIR)/main.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/main.c -o $@
#include "simd.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

static constexpr auto BUFSIZE = sizeof(simd_vector) * 4069;

struct lwcount final
{
	std::size_t lcount;
	std::size_t wcount;
	std::size_t ccount;
};

// fallback to non-seeking variant (from fastlwc.c) for non-seekable files
// (should also be used for short files less than 10MB or so)
lwcount wc(int fd)
{
	lwcount ret = {0, 0, 0};
	lcount_state lstate = LCOUNT_INITIAL;
	wcount_state wstate = WCOUNT_INITIAL;
	size_t rem = 0;
	ssize_t len;

	simd_vector* buf = reinterpret_cast<simd_vector*>(aligned_alloc(sizeof(simd_vector), BUFSIZE));
	if(!buf)
	{
		perror("fastlwc: alloc");
		exit(EXIT_FAILURE);
	}

	posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

	do
	{
		len = read(fd, (char*)buf + rem, BUFSIZE - rem);
		if(len > 0)
		{
			rem += static_cast<size_t>(len);
			ret.ccount += static_cast<size_t>(len);
		}
		else if(len == 0)
		{
			if(!rem)
				break;
			memset((char*)buf + rem, ' ', sizeof(simd_vector) - rem);
			rem = sizeof(simd_vector);
		}
		else
		{
			if(errno == EINTR)
				continue;
			perror("fastlwc: read");
			exit(EXIT_FAILURE);
		}

		simd_vector* vp = buf;
		while(rem >= sizeof(simd_vector))
		{
			ret.lcount += static_cast<size_t>(count_lines(*vp, &lstate));
			ret.wcount += static_cast<size_t>(count_words(*vp, &wstate));

			rem -= sizeof(simd_vector);
			vp++;
		}

		if(rem)	   // move rem leftover bytes to start of buf
			memmove(buf, vp, rem);
	} while(len);

	ret.lcount += count_lines_final(&lstate);
	ret.wcount += count_words_final(&wstate);

	free(buf);
	return ret;
}

// 100 lines of code and 6 levels of nesting... what's that, unreadable mess?
// self-commenting code, right?
lwcount wc_mt(int fd, off_t cur, off_t len)
{
	const size_t blocks = (static_cast<size_t>(len) + BUFSIZE - 1) / BUFSIZE;
	size_t lcount = 0;
	size_t wcount = 0;

#pragma omp parallel if(blocks > 1) reduction(+ : lcount) reduction(+ : wcount)
	{
		int tid = omp_get_thread_num(), threads = omp_get_num_threads();
		// allocate a bit extra for lookback
		char* rbuf = reinterpret_cast<char*>(
			aligned_alloc(sizeof(simd_vector), BUFSIZE + sizeof(simd_vector)));
		if(!rbuf)
		{
			perror("malloc");
			exit(EXIT_FAILURE);
		}
		char* buf = rbuf + sizeof(simd_vector);
		lcount_state lstate = LCOUNT_INITIAL;
		wcount_state wstate = WCOUNT_INITIAL;
		for(size_t i = 0, block;
			(block = i * static_cast<size_t>(threads) + static_cast<size_t>(tid)) < blocks;
			++i)
		{
			const size_t offset_end = static_cast<size_t>(cur) + (block + 1) * BUFSIZE;
			size_t bytes_left = BUFSIZE;	// last block size handled later
			ssize_t len_, rem = 0;

			if(block > 0)
			{	 // look back for whitespace for subsequent blocks
				bytes_left++;
				rem--;
			}

			do
			{
				len_ =
					pread(fd, buf + rem, bytes_left, static_cast<ssize_t>(offset_end - bytes_left));
				if(len_ > 0)
				{
					if(rem < 0 && len_ >= rem)
						wcount_state_set(&wstate, !isspace(buf[-1]));
					bytes_left -= static_cast<size_t>(len_);
					rem += len_;
				}
				else if(len_ == 0)
				{
					if(block != blocks - 1 || rem < 0)
					{
						fputs("unexpected end of file\n", stderr);
						exit(EXIT_FAILURE);
					}
					// not an error to reach EOF when processing last block
					memset(buf + rem, ' ', sizeof(simd_vector) - static_cast<size_t>(rem));
					rem = sizeof(simd_vector);
					bytes_left = 0;
				}
				else
				{	 // len < 0
					if(errno == EINTR)
						continue;
					perror("read");
					exit(EXIT_FAILURE);
				}

				simd_vector* vp = (simd_vector*)buf;
				while(rem >= (ssize_t)sizeof(simd_vector))
				{
					lcount += static_cast<size_t>(count_lines(*vp, &lstate));
					wcount += static_cast<size_t>(count_words(*vp, &wstate));

					rem -= sizeof(simd_vector);
					vp++;
				}

				if(rem)
					memmove(buf, vp, static_cast<size_t>(rem));
			} while(bytes_left);
		}

		lcount += count_lines_final(&lstate);
		wcount += count_words_final(&wstate);
		free(rbuf);
	}

	return lwcount {.lcount = lcount, .wcount = wcount, .ccount = static_cast<size_t>(len)};
}

int main(int argc, char* argv[])
{
	if(argc < 3)
	{
		int fd = STDIN_FILENO;
		if(argc > 1 && strcmp(argv[1], "-") != 0)
		{
			fd = open(argv[1], O_RDONLY);
			if(fd < 0)
			{
				fprintf(stderr, "'%s': %s\n", argv[1], strerror(errno));
				exit(EXIT_FAILURE);
			}
		}

		lwcount total {};
		off_t cur, end;

		if((cur = lseek(fd, 0, SEEK_CUR)) == -1 || (end = lseek(fd, 0, SEEK_END)) == -1)
		{
			// it seems we can't seek this file, fallback to linear read
			total = wc(fd);
		}
		else if(end > cur)
		{
			total = wc_mt(fd, cur, end - cur);
		}

		printf(" %7zu %7zu %7zu", total.lcount, total.wcount, total.ccount);
		if(argc > 1)
			printf(" \033[32m%s\033[m", argv[1]);
		putchar('\n');

		if(fd != STDIN_FILENO)
			close(fd);

		return 0;
	}

	// At least 2 files in input
	// Instead of this maybe keep a unique pointer?
	std::vector<int> asd(static_cast<std::size_t>(argc - 1));
	std::generate(asd.begin(), asd.end(), [argv, i = 1]() mutable {
		const auto fd = open(argv[i++], O_RDONLY);
		if(fd < 0)
		{
			std::fprintf(stderr, "'%s': %s\n", argv[i - 1], std::strerror(errno));
			std::exit(EXIT_FAILURE);
		}

		return fd;
	});

	lwcount total {};
	for(int fd: asd)
	{
		lwcount curr_fd_total {};
		off_t cur, end;

		if((cur = lseek(fd, 0, SEEK_CUR)) == -1 || (end = lseek(fd, 0, SEEK_END)) == -1)
		{
			// it seems we can't seek this file, fallback to linear read
			curr_fd_total = wc(fd);
		}
		else if(end > cur)
		{
			curr_fd_total = wc_mt(fd, cur, end - cur);
		}

		// Add to the total count
		total.lcount += curr_fd_total.lcount;
		total.wcount += curr_fd_total.wcount;
		total.ccount += curr_fd_total.ccount;

		std::printf(
			" %7zu %7zu %7zu", curr_fd_total.lcount, curr_fd_total.wcount, curr_fd_total.ccount);
		const auto found_iter = std::find(asd.cbegin(), asd.cend(), fd);
		if(found_iter == asd.cend())
		{
			std::fprintf(stderr, "Whoa! Something went horribly wrong\n");
			std::exit(EXIT_FAILURE);
		}

		const auto fd_index = std::distance(asd.cbegin(), found_iter) + 1;

		std::printf(" \033[1;32m%s\033[m\n", argv[fd_index]);
	}

	std::for_each(asd.cbegin(), asd.cend(), [](int fd) { close(fd); });

	std::printf(
		" %7zu %7zu %7zu \033[1;36mtotal\033[m\n", total.lcount, total.wcount, total.ccount);

	return EXIT_SUCCESS;
}


/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 *  Authors: Dalvan Griebler <dalvangriebler@gmail.com>
 *         
 *  Copyright: GNU General Public License
 *  Description: Application that counts the number of primes between 1 and N (argument [...] are optional).
 *  File Name: prime.cpp
 *  Version: 1.0 (17/07/2016)
 *  Compilation Command: g++ -std=c++1y prime.cpp -o exe
 *	Exacution Command: ./exe -h
*/

#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <getopt.h>
#include <cstring>

int prime_number ( int n ){
	int total = 0;
	for (int i = 2; i <= n; i++ ){
		int prime = 1;
		for (int j = 2; j < i; j++ ){
			if ( i % j == 0 ){
				prime = 0;
				break;
			}
		}
		// if (prime) {
		// 	std::cout << "Prime found: " << i << std::endl;
		// }
		total = total + prime;
	}
	return total;
}


int main ( int argc, char *argv[]){
	int n = 0;
	if (argc != 2){
		std::cout << "Usage: " << argv[0] << " <max n>" << std::endl;
		exit(1);
	}
	n = atoi(argv[1]);

	auto t_start = std::chrono::high_resolution_clock::now();

	int total_primes = prime_number( n );
	
	auto t_end = std::chrono::high_resolution_clock::now();

	std::cout << n << " max\t" << total_primes << " primes\t" << std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count() << "ms" << std::endl;

	return 0;
}
/******************************************************************************/
#include <iostream>
#include <vector>
#include <vector_ops.cu>

#define MAX_LINE_LENGTH 100
#define MAX_NUM_LINES 1005

#define NUM_WORDS 20
#define WORD_BLOCK_LENGTH 6
#define NUM_BLOCKS 8
#define NUM_THREADS 128

using namespace std;

const char *TARGET_FILE = "../../inputs/day1/part1";
const char *words = "ZERO__one___two___three_four__five__six___seven_eight_nine__0_____1_____2_____3_____4_____5_____6_____7_____8_____9_____";

typedef struct file_lines_ {
    const char **lines;
    int count;
} FILE_LINES;

__host__ char *get_line(FILE *f) {
    char buf[MAX_LINE_LENGTH] = {0};
    int c, ptr = 0;

    while ((c = getc(f)) != EOF && (c != '\n') && (c != '\r')) {
        buf[ptr++] = (char) c;

        if (ptr >= MAX_LINE_LENGTH) {
            perror("Line length exceeded limit.");
            exit(1);
        }
    }

    auto buf_len = ptr + 1;
    if (buf_len == 1) return nullptr;
    char *rv = (char *) malloc(sizeof(char) * buf_len);
    strcpy_s(rv, buf_len, buf);

    return rv;
}

__host__ FILE_LINES read_in_file(const char *filename) {
    FILE *f = fopen(filename, "r");

    if (f == nullptr) {
        perror("Failed to open file.");
        exit(1);
    }

    const char **lines = (const char **) malloc(sizeof(char *) * MAX_NUM_LINES);
    for (int i = 0; i < MAX_NUM_LINES; i++) lines[i] = nullptr;
    int line_ptr = 0;
    char *s;

    while ((s = get_line(f))) {
        lines[line_ptr++] = s;

        if (line_ptr >= MAX_NUM_LINES) {
            perror("Line count exceeded limit.");
            exit(1);
        }
    }

    FILE_LINES fl;
    fl.lines = lines;
    fl.count = line_ptr;

    return fl;
}

__device__ void strlen(const char *word, int *out) {
    auto i = 0;
    while (word[i] != '\0') {
        i++;
    }
    *out = i;
}

__device__ void cuda_strstr(const char *haystack, char *needle, const char **out) {
    int hs_len, ndl_len;

    strlen(haystack, &hs_len);
    strlen(needle, &ndl_len);

    for (int i = 0; i < hs_len; i++) {
        if (haystack[i] != needle[0]) continue;
        bool match = true;
        for (int j = 1; j < ndl_len; j++) {
            if ((i + j >= hs_len) || (haystack[i + j] != needle[j])) {
                match = false;
                break;
            }
        }
        if (match) {
            *out = haystack + i;
            return;
        }
    }
    *out = nullptr;
}

__device__ void cuda_last_strstr(const char *haystack, char *needle, const char **out) {
    int haystack_len, needle_len;
    strlen(haystack, &haystack_len);
    strlen(needle, &needle_len);

    const char *end = haystack + haystack_len;
    const char *idx_of_match = nullptr;
    while (end >= haystack && idx_of_match == nullptr) {
        cuda_strstr(end, needle, &idx_of_match);
        end -= 1;
    }
    *out = idx_of_match;
}

__global__ void run(const char **inp, const char *device_words, int word_count, int *out) {
    int i = (int) (blockDim.x * blockIdx.x + threadIdx.x);

    if (i >= word_count) {
        printf("[%d] Exiting due to index exceeding word count.\n", i);
        return;
    }

    const char *line = inp[i];

    auto min_offset = 99999, max_offset = -1;
    auto min_idx = -1, max_idx = -1;

    for (auto curr_idx = 0; curr_idx < NUM_WORDS; curr_idx++){
        char word[WORD_BLOCK_LENGTH];
        auto word_start = device_words + (WORD_BLOCK_LENGTH * curr_idx);
        for(int idx = 0; idx < WORD_BLOCK_LENGTH; idx++) {
            word[idx] = word_start[idx];
            if (word[idx] == '_') word[idx] = '\0';
        }

        const char *first_idx, *last_idx;
        cuda_strstr(line, word, &first_idx);
        cuda_last_strstr(line, word, &last_idx);

        if (first_idx == nullptr) {  // if there are no occurrences of needle in haystack
            continue;
        }

        // calculate the difference between pointers (a.k.a. the index of the first letter in the word)
        int first_offset = (int) (first_idx - line);
        int last_offset = (int) (last_idx - line);

        if (first_offset < min_offset) {
            min_offset = first_offset;
            min_idx = curr_idx;
        }

        if (last_offset > max_offset) {
            max_offset = last_offset;
            max_idx = curr_idx;
        }
    }

    if (min_idx == -1 && max_idx == -1) {
        printf("[%d] No numbers or number name substrings found in string %s\n", i, line);
        out[i] = 0;
        return;
    }
    auto ret_val = (min_idx % 10) * 10 + (max_idx % 10);
    out[i] = ret_val;
}

__host__ void checkpoint(const char *loc){
    cudaError_t rc = cudaDeviceSynchronize();
    if (rc == 0) rc = cudaGetLastError();  // really try and cause an error
    if (rc != cudaError::cudaSuccess) {
        std::cout << "Failed at checkpoint " << loc << ": " << cudaGetErrorString(rc) << std::endl;
        exit((int)rc);
    }
}

__host__ void cuda_day1_main(int part) {
    if (part == 1) return;

    FILE_LINES f = read_in_file(TARGET_FILE);
    std::cout << "Successfully read in file. Number of lines: " << f.count << std::endl;
    std::cout << "First three lines: " << std::endl;
    for (int i = 0; i < 3; i++) std::cout << "\t" << f.lines[i] << std::endl;

    if (NUM_BLOCKS * NUM_THREADS < f.count){
        printf("Not enough threads (have %d, need %d).\n", NUM_BLOCKS * NUM_THREADS, f.count);
        exit(0);
    }

    auto mem_size = sizeof(int) * MAX_NUM_LINES;

    char *device_words;
    cudaMalloc(&device_words, sizeof(char) * (WORD_BLOCK_LENGTH * NUM_WORDS + 1));
    cudaMemcpy(device_words, words, sizeof(char) * (WORD_BLOCK_LENGTH * NUM_WORDS + 1), cudaMemcpyHostToDevice);


    char **lines_on_device = (char **) malloc(sizeof(char*) * MAX_NUM_LINES);
    for (int i = 0; i < f.count; i++){
        cudaMalloc(&lines_on_device[i], sizeof(char) * MAX_LINE_LENGTH);
        cudaMemset(lines_on_device[i], 0, sizeof(char) * MAX_LINE_LENGTH);
        cudaMemcpy(lines_on_device[i], f.lines[i], strlen(f.lines[i]), cudaMemcpyHostToDevice);
    }

    char **device_lines;
    cudaMalloc(&device_lines, f.count * sizeof(char**));
    cudaMemcpy(device_lines, lines_on_device, f.count * sizeof(char**), cudaMemcpyHostToDevice);

    int *device_out;

    cudaMalloc(&device_out, mem_size);
    cudaMemset(device_out, 0, mem_size);

    run<<<NUM_BLOCKS, NUM_THREADS>>>((const char **)device_lines, device_words, f.count, device_out);
    checkpoint("post_kernel_run");

    int *device_final_sum;
    cudaMalloc(&device_final_sum, sizeof(int));
    vec_reduce_sum<<<NUM_BLOCKS, NUM_THREADS>>>(f.count, device_out, device_final_sum);

    int final_sum;
    cudaMemcpy(&final_sum, device_final_sum, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << final_sum << std::endl;
}

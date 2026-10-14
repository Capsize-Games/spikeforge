/* Stub nir-c-runtime for the ctypes binding tests.
 *
 * Implements the eight C functions spikeforge's binding calls, with the
 * same signatures, so the tests exercise the real marshaling: struct
 * layouts, the error-string protocol, and the load/run/reset/close
 * lifecycle. The tensor/graph/runtime layouts below mirror
 * nir-c-runtime's include/nir_c_runtime/{tensor,graph,runtime}.h at the
 * commit spikeforge_targets tracks (see TRACKED_COMMIT in
 * spikeforge_targets/c_runtime.py); the stub verifies the tensor
 * fields Python writes, so a layout drift fails here, not silently.
 *
 * Behavior is scenario-driven by markers in the parsed text, plus one
 * global step counter standing in for carried runtime state:
 *
 * - "STUB_PARSE_FAIL" in the text: json_parse fails.
 * - "STUB_LOAD_FAIL" in the text: nir_graph_load fails.
 * - "STUB_INIT_FAIL" in the text: nir_runtime_init fails.
 * - a NaN input sample: nir_runtime_run fails.
 *
 * Otherwise each run returns [steps, steps + 1] with steps counting
 * the runs since the last init, proving state carries across steps
 * and reset clears it. Every load re-initializes, so each test that
 * loads its own graph starts from a deterministic counter.
 */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *data;
    size_t ndim;
    size_t shape[3];
    size_t size;
    size_t capacity;
    int owns_data;
} stub_tensor_t;

typedef struct {
    void *nodes;
    size_t node_count;
    void *edges;
    size_t edge_count;
    size_t input_index;
    size_t output_index;
} stub_graph_t;

typedef struct {
    const void *graph;
    void *order;
    void *li_state;
    void *li_primed;
    void *delay_buffer;
    void *delay_primed;
} stub_runtime_t;

static int stub_steps = 0;

static void set_error(char **error, const char *message) {
    if (error == NULL || *error != NULL) {
        return;
    }
    *error = (char *)malloc(strlen(message) + 1);
    if (*error != NULL) {
        strcpy(*error, message);
    }
}

void *json_parse(const char *text, char **error) {
    if (strstr(text, "STUB_PARSE_FAIL") != NULL) {
        set_error(error, "stub: bad json");
        return NULL;
    }
    if (strstr(text, "STUB_LOAD_FAIL") != NULL) {
        return (void *)2;
    }
    if (strstr(text, "STUB_INIT_FAIL") != NULL) {
        return (void *)3;
    }
    return (void *)1;
}

void json_free(void *value) {
    (void)value;
}

int nir_graph_load(
    const void *summary, stub_graph_t *out, char **error
) {
    memset(out, 0, sizeof(*out));
    if (summary == (const void *)2) {
        set_error(error, "stub: unsupported kind 'Nope'");
        return 0;
    }
    out->node_count = (summary == (const void *)3) ? 0 : 1;
    return 1;
}

void nir_graph_free(stub_graph_t *graph) {
    memset(graph, 0, sizeof(*graph));
}

int nir_runtime_init(
    stub_runtime_t *rt, const stub_graph_t *graph, char **error
) {
    memset(rt, 0, sizeof(*rt));
    if (graph->node_count == 0) {
        set_error(error, "stub: bad init");
        return 0;
    }
    rt->graph = graph;
    stub_steps = 0;
    return 1;
}

void nir_runtime_free(stub_runtime_t *rt) {
    memset(rt, 0, sizeof(*rt));
}

int nir_runtime_run(
    stub_runtime_t *rt, const stub_tensor_t *frames, size_t num_steps,
    stub_tensor_t *readout_out, char **error
) {
    (void)rt;
    const stub_tensor_t *frame;
    float *data;
    size_t expect, i;
    if (num_steps != 1) {
        set_error(error, "stub: want exactly 1 step");
        return 0;
    }
    frame = &frames[0];
    if (frame->data == NULL || frame->ndim < 1 || frame->ndim > 3 ||
        frame->size == 0) {
        set_error(error, "stub: bad input tensor");
        return 0;
    }
    expect = 1;
    for (i = 0; i < frame->ndim; i++) {
        expect *= frame->shape[i];
    }
    if (expect != frame->size || frame->capacity < frame->size) {
        set_error(error, "stub: shape/size mismatch");
        return 0;
    }
    for (i = 0; i < frame->size; i++) {
        if (frame->data[i] != frame->data[i]) {
            set_error(error, "stub: nan input");
            return 0;
        }
    }
    stub_steps += 1;
    data = (float *)malloc(2 * sizeof(float));
    if (data == NULL) {
        set_error(error, "out of memory");
        return 0;
    }
    data[0] = (float)stub_steps;
    data[1] = (float)stub_steps + 1.0f;
    memset(readout_out, 0, sizeof(*readout_out));
    readout_out->data = data;
    readout_out->ndim = 1;
    readout_out->shape[0] = 2;
    readout_out->size = 2;
    readout_out->capacity = 2;
    readout_out->owns_data = 1;
    return 1;
}

void nir_tensor_free(stub_tensor_t *tensor) {
    if (tensor->owns_data) {
        free(tensor->data);
    }
    memset(tensor, 0, sizeof(*tensor));
}

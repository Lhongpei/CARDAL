/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Hongpei Li
 */

#include "generator.h"
#include "utils.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

basic_sdp_t *generate_order_sdp(int n_size, int k) {
    LOG_DBG("Generating Order SDP for N=%d, k=%d...\n", n_size, k);

    int num_blocks = k - 1; // Drop Q_0 and Q_k 
    int num_translation_constr = k * (n_size - 1); // 
    int num_trace_constr = k - 1; // 
    int m_total = num_translation_constr + num_trace_constr; // 

    basic_sdp_t *input = (basic_sdp_t *)safe_malloc(sizeof(basic_sdp_t));
    input->m = m_total;
    input->n_cones = num_blocks;
    
    input->blk_dims = (int *)safe_malloc(num_blocks * sizeof(int));
    for (int i = 0; i < num_blocks; i++) {
        input->blk_dims[i] = n_size;
    }

    input->right_hand_side = (double *)calloc(m_total, sizeof(double));

    // Estimate capacity for constraint non-zeros to minimize reallocs
    int capacity = num_blocks * n_size * n_size * 2; 
    int a_nnz = 0;
    
    int *a_constr = (int *)safe_malloc(capacity * sizeof(int));
    int *a_cone   = (int *)safe_malloc(capacity * sizeof(int));
    int *a_row    = (int *)safe_malloc(capacity * sizeof(int));
    int *a_col    = (int *)safe_malloc(capacity * sizeof(int));
    double *a_val = (double *)safe_malloc(capacity * sizeof(double));

    // Helper macro to append matrix entries dynamically
    #define ADD_CONSTR(c_idx, cone, r, c, v) do { \
        if (a_nnz >= capacity) { \
            capacity = capacity * 2 + 1000; \
            a_constr = (int *)realloc(a_constr, capacity * sizeof(int)); \
            a_cone   = (int *)realloc(a_cone, capacity * sizeof(int)); \
            a_row    = (int *)realloc(a_row, capacity * sizeof(int)); \
            a_col    = (int *)realloc(a_col, capacity * sizeof(int)); \
            a_val    = (double *)realloc(a_val, capacity * sizeof(double)); \
        } \
        a_constr[a_nnz] = (c_idx); \
        a_cone[a_nnz]   = (cone); \
        a_row[a_nnz]    = (r); \
        a_col[a_nnz]    = (c); \
        a_val[a_nnz]    = (v); \
        a_nnz++; \
    } while(0)

    for (int t = 1; t <= k; t++) {
        
        for (int d = 1; d <= n_size - 1; d++) {
            int constr_idx = (t - 1) * (n_size - 1) + (d - 1); 
            double sign_t = (t % 2 != 0) ? -1.0 : 1.0;

            if (t == 1) {
                int curr_cone = 0;
                
                for (int i = 1; i <= n_size - d; i++) {
                    int j = i + d;
                    ADD_CONSTR(constr_idx, curr_cone, i - 1, j - 1, 0.5);
                    ADD_CONSTR(constr_idx, curr_cone, j - 1, i - 1, 0.5);
                }
                
                for (int i = 1; i <= d; i++) {
                    int j = i + (n_size - d);
                    ADD_CONSTR(constr_idx, curr_cone, j - 1, i - 1, sign_t * 0.5);
                    ADD_CONSTR(constr_idx, curr_cone, i - 1, j - 1, sign_t * 0.5);
                }

                double b_val = (double)(n_size - d) / n_size;
                b_val = b_val - (double)d / n_size;
                input->right_hand_side[constr_idx] = b_val;

            } else if (t == k) {
                int prev_cone = k - 2;
                
                for (int i = 1; i <= n_size - d; i++) {
                    int j = i + d;
                    ADD_CONSTR(constr_idx, prev_cone, i - 1, j - 1, -0.5);
                    ADD_CONSTR(constr_idx, prev_cone, j - 1, i - 1, -0.5);
                }
                
                for (int i = 1; i <= d; i++) {
                    int j = i + (n_size - d);
                    ADD_CONSTR(constr_idx, prev_cone, j - 1, i - 1, -sign_t * 0.5);
                    ADD_CONSTR(constr_idx, prev_cone, i - 1, j - 1, -sign_t * 0.5);
                }

            } else {
                int curr_cone = t - 1;
                int prev_cone = t - 2;
                
                for (int i = 1; i <= n_size - d; i++) {
                    int j = i + d;
                    
                    ADD_CONSTR(constr_idx, curr_cone, i - 1, j - 1, 0.5);
                    ADD_CONSTR(constr_idx, curr_cone, j - 1, i - 1, 0.5);
                    
                    ADD_CONSTR(constr_idx, prev_cone, i - 1, j - 1, -0.5);
                    ADD_CONSTR(constr_idx, prev_cone, j - 1, i - 1, -0.5);
                }
                
                for (int i = 1; i <= d; i++) {
                    int j = i + (n_size - d);
                    
                    ADD_CONSTR(constr_idx, curr_cone, j - 1, i - 1, sign_t * 0.5);
                    ADD_CONSTR(constr_idx, curr_cone, i - 1, j - 1, sign_t * 0.5);
                    
                    ADD_CONSTR(constr_idx, prev_cone, j - 1, i - 1, -sign_t * 0.5); 
                    ADD_CONSTR(constr_idx, prev_cone, i - 1, j - 1, -sign_t * 0.5); 
                }
            }
        }

        if (t <= k - 1) {
            int constr_idx = num_translation_constr + (t - 1);
            int curr_cone = t - 1;
            
            for (int i = 1; i <= n_size; i++) {
                ADD_CONSTR(constr_idx, curr_cone, i - 1, i - 1, 1.0);
            }
            input->right_hand_side[constr_idx] = 1.0;
        }
    }

    #undef ADD_CONSTR

    input->nnz_psd_constr = a_nnz;
    input->psd_cone_constraints = (psd_cone_constraint_t *)safe_malloc(sizeof(psd_cone_constraint_t));
    input->psd_cone_constraints->constr_ind = a_constr;
    input->psd_cone_constraints->cone_ind   = a_cone;
    input->psd_cone_constraints->row_ind    = a_row;
    input->psd_cone_constraints->col_ind    = a_col;
    input->psd_cone_constraints->val        = a_val;

    input->nnz_psd_obj = 0; 
    input->psd_cone_objective = NULL;

    input->lp_constraints = NULL;
    input->lp_objective = NULL;
    input->nnz_lp_constr = 0;
    input->nnz_lp_obj = 0;
    input->lp_dim = 0;

    return input;
}
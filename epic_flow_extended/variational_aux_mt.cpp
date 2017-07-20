#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <algorithm>
#include <stdexcept>

#include "variational_aux_mt.h"

#include "../penalty_functions/quadratic_function.h"
#include "../penalty_functions/trunc_modified_l1_norm.h"
#include "../penalty_functions/lorentzian.h"
#include "../penalty_functions/geman_mcclure.h"

#define RECTIFY(a,b) (((a)<0) ? (0) : ( ((a)<(b)-1) ? (a) : ((b)-1) ) )

void Variational_AUX_MT::compute_smoothness(int method, image_t *dst_horiz, image_t *dst_vert, const image_t *uu, const image_t *vv, const image_t *dpsis_weight, const image_t *dpsis_weight_x,
		const image_t *dpsis_weight_y, const convolution_t *deriv_flow, const float alpha) const {
	int w = uu->width, h = uu->height, s = uu->stride, i, j, offset;
	image_t *ux1 = image_new(w, h), *uy1 = image_new(w, h), *vx1 = image_new(w, h), *vy1 = image_new(w, h), *ux2 = image_new(w, h), *uy2 = image_new(w, h), *vx2 = image_new(w, h), *vy2 = image_new(w,
			h);
	// compute ux1, vx1, filter [-1 1]
	for (j = 0; j < h; j++) {
		offset = j * s;
		for (i = 0; i < w - 1; i++, offset++) {
			ux1->data[offset] = uu->data[offset + 1] - uu->data[offset];
			vx1->data[offset] = vv->data[offset + 1] - vv->data[offset];
		}
	}
	// compute uy1, vy1, filter [-1;1]
	for (j = 0; j < h - 1; j++) {
		offset = j * s;
		for (i = 0; i < w; i++, offset++) {
			uy1->data[offset] = uu->data[offset + s] - uu->data[offset];
			vy1->data[offset] = vv->data[offset + s] - vv->data[offset];
		}
	}

	// compute ux2, uy2, vx2, vy2, filter [-0.5 0 0.5]
	convolve_horiz(ux2, uu, deriv_flow);
	convolve_horiz(vx2, vv, deriv_flow);
	convolve_vert(uy2, uu, deriv_flow);
	convolve_vert(vy2, vv, deriv_flow);
	if (method <= 1) {
		float tmp = 0, tmp2 = 0, tmp_w = 0;

		// compute final value, horiz
		for (j = 0; j < h; j++) {
			offset = j * s;
			for (i = 0; i < w - 1; i++, offset++) {
				tmp = 0, tmp2 = 0;
				tmp_w = (dpsis_weight->data[offset] + dpsis_weight->data[offset + 1]);

				if (method == 1) {
					tmp_w = (dpsis_weight->data[offset] + dpsis_weight->data[offset + 1]);
					tmp = 0.5f * (uy2->data[offset] + uy2->data[offset + 1]);
					tmp2 = 0.5f * (vy2->data[offset] + vy2->data[offset + 1]);
				}

				tmp = ux1->data[offset] * ux1->data[offset] + tmp * tmp; // uxsq
				tmp2 = vx1->data[offset] * vx1->data[offset] + tmp2 * tmp2; // vxsq

				tmp = tmp + tmp2;

				dst_horiz->data[offset] = tmp_w * alpha * robust_reg->derivative(tmp);
			}
			memset(&dst_horiz->data[j * s + w - 1], 0, sizeof(float) * (s - w + 1)); // constant border
		}

		// compute final value, vert
		for (j = 0; j < h - 1; j++) {
			offset = j * s;
			for (i = 0; i < w; i++, offset++) {
				tmp = 0, tmp2 = 0;
				tmp_w = (dpsis_weight->data[offset] + dpsis_weight->data[offset + s]);

				if (method == 1) {
					tmp_w = (dpsis_weight->data[offset] + dpsis_weight->data[offset + s]);
					tmp = 0.5f * (ux2->data[offset] + ux2->data[offset + s]);
					tmp2 = 0.5f * (vx2->data[offset] + vx2->data[offset + s]);
				}

				tmp = uy1->data[offset] * uy1->data[offset] + tmp * tmp; // uysq
				tmp2 = vy1->data[offset] * vy1->data[offset] + tmp2 * tmp2; // vysq

				tmp = tmp + tmp2;

				dst_vert->data[offset] = tmp_w * alpha * robust_reg->derivative(tmp);
			}
		}
		memset(&dst_vert->data[(h - 1) * s], 0, sizeof(float) * s); // constant border

	} else {
		// compute final weight
		for (j = 0; j < h; j++) {
			offset = j * s;
			for (i = 0; i < w; i++, offset++) {
				float tmp = 0;
				float w = dpsis_weight->data[offset];

				// constant border condition
				if (i < w - 1) {
					tmp += ux1->data[offset] * ux1->data[offset] + vx1->data[offset] * vx1->data[offset];
					w += dpsis_weight->data[offset + 1];
				}
				if (j < h - 1) {
					tmp += vy1->data[offset] * vy1->data[offset] + uy1->data[offset] * uy1->data[offset];
					w += dpsis_weight->data[offset + s];
				}

				dst_horiz->data[offset] = w * alpha * robust_reg->derivative(tmp);

				dst_vert->data[offset] = dst_horiz->data[offset]; // we use the same weighting for the vertical and horizontal smoothness
			}
		}
	}

	image_delete(ux1);
	image_delete(uy1);
	image_delete(vx1);
	image_delete(vy1);
	image_delete(ux2);
	image_delete(uy2);
	image_delete(vx2);
	image_delete(vy2);
}

/* sub the laplacian (smoothness term) to the right-hand term */
void Variational_AUX_MT::sub_laplacian(image_t *dst, const image_t *src, const image_t *weight_horiz, const image_t *weight_vert) {
	int j;
	const int offsetline = src->stride - src->width;
	float *src_ptr = src->data, *dst_ptr = dst->data, *weight_horiz_ptr = weight_horiz->data;
	// horizontal filtering
	for (j = src->height + 1; --j;) { // faster than for(j=0;j<src->height;j++)
		int i;
		for (i = src->width; --i;) {
			const float tmp = (*weight_horiz_ptr) * ((*(src_ptr + 1)) - (*src_ptr)); // inner derivative of robust fct
			*dst_ptr += tmp; // plus value at i,j
			*(dst_ptr + 1) -= tmp; // minus value at i,j+1
			dst_ptr++;
			src_ptr++;
			weight_horiz_ptr++;
		}
		dst_ptr += offsetline + 1;
		src_ptr += offsetline + 1;
		weight_horiz_ptr += offsetline + 1;
	}

	v4sf *wvp = (v4sf*) weight_vert->data, *srcp = (v4sf*) src->data, *srcp_s = (v4sf*) (src->data + src->stride), *dstp = (v4sf*) dst->data, *dstp_s = (v4sf*) (dst->data + src->stride);
	for (j = 1 + (src->height - 1) * src->stride / 4; --j;) {
		const v4sf tmp = (*wvp) * ((*srcp_s) - (*srcp)); // inner derivative of robust fct
		*dstp += tmp; // plus value at i,j
		*dstp_s -= tmp; // minus value at i+1,j
		wvp += 1;
		srcp += 1;
		srcp_s += 1;
		dstp += 1;
		dstp_s += 1;
	}
}

/* compute the dataterm and the matching term
 a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
 other (color) images are input */
void Variational_AUX_MT::add_data_and_match(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *du, image_t *dv, color_image_t **Ix, color_image_t **Iy,
		color_image_t **Iz, color_image_t **Ixx, color_image_t **Ixy, color_image_t **Iyy, color_image_t **Ixz, color_image_t **Iyz, const float delta_over3, const float gamma_over3, int f,
		const float s) const {

	const v4sf dnorm = { datanorm, datanorm, datanorm, datanorm };
	const v4sf hdover3 = { delta_over3, delta_over3, delta_over3, delta_over3 };
	const v4sf hgover3 = { gamma_over3, gamma_over3, gamma_over3, gamma_over3 };

	const v4sf factor = { s, s, s, s };
	const v4sf factorp1 = { s + 1, s + 1, s + 1, s + 1 };

	v4sf *dup = (v4sf*) du->data, *dvp = (v4sf*) dv->data, *maskp = (v4sf*) mask->data, *weight1p = (v4sf*) channel_w->c1, *weight2p = (v4sf*) channel_w->c2, *weight3p = (v4sf*) channel_w->c3, *a11p =
			(v4sf*) a11->data, *a12p = (v4sf*) a12->data, *a22p = (v4sf*) a22->data, *b1p = (v4sf*) b1->data, *b2p = (v4sf*) b2->data, *iz1p = (v4sf*) Iz[f]->c1, *ixz1p = (v4sf*) Ixz[f]->c1, *iyz1p =
			(v4sf*) Iyz[f]->c1, *ix1p = (v4sf*) Ix[f]->c1, *iy1p = (v4sf*) Iy[f]->c1, *ixx1p = (v4sf*) Ixx[f]->c1, *ixy1p = (v4sf*) Ixy[f]->c1, *iyy1p = (v4sf*) Iyy[f]->c1, *iz2p = (v4sf*) Iz[f]->c2,
			*ixz2p = (v4sf*) Ixz[f]->c2, *iyz2p = (v4sf*) Iyz[f]->c2, *ix2p = (v4sf*) Ix[f]->c2, *iy2p = (v4sf*) Iy[f]->c2, *ixx2p = (v4sf*) Ixx[f]->c2, *ixy2p = (v4sf*) Ixy[f]->c2, *iyy2p =
					(v4sf*) Iyy[f]->c2, *iz3p = (v4sf*) Iz[f]->c3, *ixz3p = (v4sf*) Ixz[f]->c3, *iyz3p = (v4sf*) Iyz[f]->c3, *ix3p = (v4sf*) Ix[f]->c3, *iy3p = (v4sf*) Iy[f]->c3, *ixx3p =
					(v4sf*) Ixx[f]->c3, *ixy3p = (v4sf*) Ixy[f]->c3, *iyy3p = (v4sf*) Iyy[f]->c3;

	v4sf tmp, tmp2, tmp3, tmp4, tmp5, tmp6, tmp_ix, tmp_iy, tmp_ixy, tmp2_ix, tmp2_iy, tmp2_ixy, tmp3_ix, tmp3_iy, tmp3_ixy, n1, n2, n3, n4, n5, n6;

	int i;
	for (i = 0; i < du->height * du->stride / 4; i++) {
		// dpsi color
		if (delta_over3) {
			tmp = (*weight1p) * (*iz1p + (*ix1p) * factor * (*dup) + (*iy1p) * factor * (*dvp) - (*ix1p) * factorp1 * (*dup) - (*iy1p) * factorp1 * (*dvp));
			tmp2 = (*weight2p) * (*iz2p + (*ix2p) * factor * (*dup) + (*iy2p) * factor * (*dvp) - (*ix2p) * factorp1 * (*dup) - (*iy2p) * factorp1 * (*dvp));
			tmp3 = (*weight3p) * (*iz3p + (*ix3p) * factor * (*dup) + (*iy3p) * factor * (*dvp) - (*ix3p) * factorp1 * (*dup) - (*iy3p) * factorp1 * (*dvp));

			if (!dt_norm) {
				// without NORMALIZATION
				tmp = (*maskp) * hdover3 * robust_color->derivative(tmp * tmp + tmp2 * tmp2 + tmp3 * tmp3);

				tmp_ix = factor * (*ix1p) - factorp1 * (*ix1p);
				tmp_iy = factor * (*iy1p) - factorp1 * (*iy1p);

				tmp2 = tmp * (*weight1p);
				*a11p += tmp2 * (tmp_ix) * (tmp_ix);
				*a12p += tmp2 * (tmp_ix) * (tmp_iy);
				*a22p += tmp2 * (tmp_iy) * (tmp_iy);
				*b1p -= tmp2 * (*iz1p) * (tmp_ix);
				*b2p -= tmp2 * (*iz1p) * (tmp_iy);

				tmp_ix = factor * (*ix2p) - factorp1 * (*ix2p);
				tmp_iy = factor * (*iy2p) - factorp1 * (*iy2p);

				tmp2 = tmp * (*weight2p);
				*a11p += tmp2 * (tmp_ix) * (tmp_ix);
				*a12p += tmp2 * (tmp_ix) * (tmp_iy);
				*a22p += tmp2 * (tmp_iy) * (tmp_iy);
				*b1p -= tmp2 * (*iz2p) * (tmp_ix);
				*b2p -= tmp2 * (*iz2p) * (tmp_iy);

				tmp_ix = factor * (*ix3p) - factorp1 * (*ix3p);
				tmp_iy = factor * (*iy3p) - factorp1 * (*iy3p);

				tmp2 = tmp * (*weight3p);
				*a11p += tmp2 * (tmp_ix) * (tmp_ix);
				*a12p += tmp2 * (tmp_ix) * (tmp_iy);
				*a22p += tmp2 * (tmp_iy) * (tmp_iy);
				*b1p -= tmp2 * (*iz3p) * (tmp_ix);
				*b2p -= tmp2 * (*iz3p) * (tmp_iy);
			} else {
				// with NORMALIZATION
				tmp_ix = factor * (*ix1p) - factorp1 * (*ix1p);
				tmp_iy = factor * (*iy1p) - factorp1 * (*iy1p);
				tmp2_ix = factor * (*ix2p) - factorp1 * (*ix2p);
				tmp2_iy = factor * (*iy2p) - factorp1 * (*iy2p);
				tmp3_ix = factor * (*ix3p) - factorp1 * (*ix3p);
				tmp3_iy = factor * (*iy3p) - factorp1 * (*iy3p);

				n1 = (tmp_ix) * (tmp_ix) + (tmp_iy) * (tmp_iy) + dnorm;
				n2 = (tmp2_ix) * (tmp2_ix) + (tmp2_iy) * (tmp2_iy) + dnorm;
				n3 = (tmp3_ix) * (tmp3_ix) + (tmp3_iy) * (tmp3_iy) + dnorm;

				tmp = (*maskp) * hdover3 * robust_color->derivative(tmp * tmp / n1 + tmp2 * tmp2 / n2 + tmp3 * tmp3 / n3);
				tmp3 = tmp / n3;
				tmp2 = tmp / n2;
				tmp /= n1;

				tmp = tmp * (*weight1p);
				*a11p += tmp * (tmp_ix) * (tmp_ix);
				*a12p += tmp * (tmp_ix) * (tmp_iy);
				*a22p += tmp * (tmp_iy) * (tmp_iy);
				*b1p -= tmp * (*iz1p) * (tmp_ix);
				*b2p -= tmp * (*iz1p) * (tmp_iy);

				tmp2 = tmp2 * (*weight2p);
				*a11p += tmp2 * (tmp2_ix) * (tmp2_ix);
				*a12p += tmp2 * (tmp2_ix) * (tmp2_iy);
				*a22p += tmp2 * (tmp2_iy) * (tmp2_iy);
				*b1p -= tmp2 * (*iz2p) * (tmp2_ix);
				*b2p -= tmp2 * (*iz2p) * (tmp2_iy);

				tmp3 = tmp3 * (*weight3p);
				*a11p += tmp3 * (tmp3_ix) * (tmp3_ix);
				*a12p += tmp3 * (tmp3_ix) * (tmp3_iy);
				*a22p += tmp3 * (tmp3_iy) * (tmp3_iy);
				*b1p -= tmp3 * (*iz3p) * (tmp3_ix);
				*b2p -= tmp3 * (*iz3p) * (tmp3_iy);
			}
		}

		// dpfactori gradient
		tmp = (*weight1p) * (*ixz1p + (*ixx1p) * factor * (*dup) + (*ixy1p) * factor * (*dvp) - (*ixx1p) * factorp1 * (*dup) - (*ixy1p) * factorp1 * (*dvp));
		tmp2 = (*weight1p) * (*iyz1p + (*ixy1p) * factor * (*dup) + (*iyy1p) * factor * (*dvp) - (*ixy1p) * factorp1 * (*dup) - (*iyy1p) * factorp1 * (*dvp));

		tmp3 = (*weight2p) * (*ixz2p + (*ixx2p) * factor * (*dup) + (*ixy2p) * factor * (*dvp) - (*ixx2p) * factorp1 * (*dup) - (*ixy2p) * factorp1 * (*dvp));
		tmp4 = (*weight2p) * (*iyz2p + (*ixy2p) * factor * (*dup) + (*iyy2p) * factor * (*dvp) - (*ixy2p) * factorp1 * (*dup) - (*iyy2p) * factorp1 * (*dvp));

		tmp5 = (*weight3p) * (*ixz3p + (*ixx3p) * factor * (*dup) + (*ixy3p) * factor * (*dvp) - (*ixx3p) * factorp1 * (*dup) - (*ixy3p) * factorp1 * (*dvp));
		tmp6 = (*weight3p) * (*iyz3p + (*ixy3p) * factor * (*dup) + (*iyy3p) * factor * (*dvp) - (*ixy3p) * factorp1 * (*dup) - (*iyy3p) * factorp1 * (*dvp));

		if (!dt_norm) {
			// without NORMALIZATION
			tmp = (*maskp) * hgover3 * robust_grad->derivative(tmp * tmp + tmp2 * tmp2 + tmp3 * tmp3 + tmp4 * tmp4 + tmp5 * tmp5 + tmp6 * tmp6);

			tmp_ix = factor * (*ixx1p) - factorp1 * (*ixx1p);
			tmp_iy = factor * (*iyy1p) - factorp1 * (*iyy1p);
			tmp_ixy = factor * (*ixy1p) - factorp1 * (*ixy1p);

			tmp2 = tmp * (*weight1p);
			*a11p += tmp2 * (tmp_ix) * (tmp_ix) + tmp2 * (tmp_ixy) * (tmp_ixy);
			*a12p += tmp2 * (tmp_ix) * (tmp_ixy) + tmp2 * (tmp_ixy) * (tmp_iy);
			*a22p += tmp2 * (tmp_iy) * (tmp_iy) + tmp2 * (tmp_ixy) * (tmp_ixy);
			*b1p -= tmp2 * (*ixz1p) * (tmp_ix) + tmp2 * (*iyz1p) * (tmp_ixy);
			*b2p -= tmp2 * (*iyz1p) * (tmp_iy) + tmp2 * (*ixz1p) * (tmp_ixy);

			tmp_ix = factor * (*ixx2p) - factorp1 * (*ixx2p);
			tmp_iy = factor * (*iyy2p) - factorp1 * (*iyy2p);
			tmp_ixy = factor * (*ixy2p) - factorp1 * (*ixy2p);

			tmp2 = tmp * (*weight2p);
			*a11p += tmp2 * (tmp_ix) * (tmp_ix) + tmp2 * (tmp_ixy) * (tmp_ixy);
			*a12p += tmp2 * (tmp_ix) * (tmp_ixy) + tmp2 * (tmp_ixy) * (tmp_iy);
			*a22p += tmp2 * (tmp_iy) * (tmp_iy) + tmp2 * (tmp_ixy) * (tmp_ixy);
			*b1p -= tmp2 * (*ixz2p) * (tmp_ix) + tmp2 * (*iyz2p) * (tmp_ixy);
			*b2p -= tmp2 * (*iyz2p) * (tmp_iy) + tmp2 * (*ixz2p) * (tmp_ixy);

			tmp_ix = factor * (*ixx3p) - factorp1 * (*ixx3p);
			tmp_iy = factor * (*iyy3p) - factorp1 * (*iyy3p);
			tmp_ixy = factor * (*ixy3p) - factorp1 * (*ixy3p);

			tmp2 = tmp * (*weight3p);
			*a11p += tmp2 * (tmp_ix) * (tmp_ix) + tmp2 * (tmp_ixy) * (tmp_ixy);
			*a12p += tmp2 * (tmp_ix) * (tmp_ixy) + tmp2 * (tmp_ixy) * (tmp_iy);
			*a22p += tmp2 * (tmp_iy) * (tmp_iy) + tmp2 * (tmp_ixy) * (tmp_ixy);
			*b1p -= tmp2 * (*ixz3p) * (tmp_ix) + tmp2 * (*iyz3p) * (tmp_ixy);
			*b2p -= tmp2 * (*iyz3p) * (tmp_iy) + tmp2 * (*ixz3p) * (tmp_ixy);
		} else {
			// with NORMALIZATION
			tmp_ix = factor * (*ixx1p) - factorp1 * (*ixx1p);
			tmp_iy = factor * (*iyy1p) - factorp1 * (*iyy1p);
			tmp_ixy = factor * (*ixy1p) - factorp1 * (*ixy1p);
			tmp2_ix = factor * (*ixx2p) - factorp1 * (*ixx2p);
			tmp2_iy = factor * (*iyy2p) - factorp1 * (*iyy2p);
			tmp2_ixy = factor * (*ixy2p) - factorp1 * (*ixy2p);
			tmp3_ix = factor * (*ixx3p) - factorp1 * (*ixx3p);
			tmp3_iy = factor * (*iyy3p) - factorp1 * (*iyy3p);
			tmp3_ixy = factor * (*ixy3p) - factorp1 * (*ixy3p);

			n1 = (tmp_ix) * (tmp_ix) + (tmp_ixy) * (tmp_ixy) + dnorm;
			n2 = (tmp_iy) * (tmp_iy) + (tmp_ixy) * (tmp_ixy) + dnorm;
			n3 = (tmp2_ix) * (tmp2_ix) + (tmp2_ixy) * (tmp2_ixy) + dnorm;
			n4 = (tmp2_iy) * (tmp2_iy) + (tmp2_ixy) * (tmp2_ixy) + dnorm;
			n5 = (tmp3_ix) * (tmp3_ix) + (tmp3_ixy) * (tmp3_ixy) + dnorm;
			n6 = (tmp3_iy) * (tmp3_iy) + (tmp3_ixy) * (tmp3_ixy) + dnorm;

			tmp = (*maskp) * hgover3 * robust_grad->derivative(tmp * tmp / n1 + tmp2 * tmp2 / n2 + tmp3 * tmp3 / n3 + tmp4 * tmp4 / n4 + tmp5 * tmp5 / n5 + tmp6 * tmp6 / n6);
			tmp6 = tmp / n6;
			tmp5 = tmp / n5;
			tmp4 = tmp / n4;
			tmp3 = tmp / n3;
			tmp2 = tmp / n2;
			tmp /= n1;

			tmp = tmp * (*weight1p);
			tmp2 = tmp2 * (*weight1p);
			*a11p += tmp * (tmp_ix) * (tmp_ix) + tmp2 * (tmp_ixy) * (tmp_ixy);
			*a12p += tmp * (tmp_ix) * (tmp_ixy) + tmp2 * (tmp_ixy) * (tmp_iy);
			*a22p += tmp2 * (tmp_iy) * (tmp_iy) + tmp * (tmp_ixy) * (tmp_ixy);
			*b1p -= tmp * (*ixz1p) * (tmp_ix) + tmp2 * (*iyz1p) * (tmp_ixy);
			*b2p -= tmp2 * (*iyz1p) * (tmp_iy) + tmp * (*ixz1p) * (tmp_ixy);

			tmp3 = tmp3 * (*weight2p);
			tmp4 = tmp4 * (*weight2p);
			*a11p += tmp3 * (tmp2_ix) * (tmp2_ix) + tmp4 * (tmp2_ixy) * (tmp2_ixy);
			*a12p += tmp3 * (tmp2_ix) * (tmp2_ixy) + tmp4 * (tmp2_ixy) * (tmp2_iy);
			*a22p += tmp4 * (tmp2_iy) * (tmp2_iy) + tmp3 * (tmp2_ixy) * (tmp2_ixy);
			*b1p -= tmp3 * (*ixz2p) * (tmp2_ix) + tmp4 * (*iyz2p) * (tmp2_ixy);
			*b2p -= tmp4 * (*iyz2p) * (tmp2_iy) + tmp3 * (*ixz2p) * (tmp2_ixy);

			tmp5 = tmp5 * (*weight3p);
			tmp6 = tmp6 * (*weight3p);
			*a11p += tmp5 * (tmp3_ix) * (tmp3_ix) + tmp6 * (tmp3_ixy) * (tmp3_ixy);
			*a12p += tmp5 * (tmp3_ix) * (tmp3_ixy) + tmp6 * (tmp3_ixy) * (tmp3_iy);
			*a22p += tmp6 * (tmp3_iy) * (tmp3_iy) + tmp5 * (tmp3_ixy) * (tmp3_ixy);
			*b1p -= tmp5 * (*ixz3p) * (tmp3_ix) + tmp6 * (*iyz3p) * (tmp3_ixy);
			*b2p -= tmp6 * (*iyz3p) * (tmp3_iy) + tmp5 * (*ixz3p) * (tmp3_ixy);
		}

		dup += 1;
		dvp += 1;
		maskp += 1;
		weight1p += 1;
		weight2p += 1;
		weight3p += 1;
		a11p += 1;
		a12p += 1;
		a22p += 1;
		b1p += 1;
		b2p += 1;

		ix1p += 1;
		iy1p += 1;
		iz1p += 1;
		ixx1p += 1;
		ixy1p += 1;
		iyy1p += 1;
		ixz1p += 1;
		iyz1p += 1;
		ix2p += 1;
		iy2p += 1;
		iz2p += 1;
		ixx2p += 1;
		ixy2p += 1;
		iyy2p += 1;
		ixz2p += 1;
		iyz2p += 1;
		ix3p += 1;
		iy3p += 1;
		iz3p += 1;
		ixx3p += 1;
		ixy3p += 1;
		iyy3p += 1;
		ixz3p += 1;
		iyz3p += 1;
	}
}

/* compute the dataterm and the matching term
 a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
 other (color) images are input */
void Variational_AUX_MT::add_data_and_match_ref(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *du, image_t *dv, color_image_t **Ix, color_image_t **Iy,
		color_image_t **Iz, color_image_t **Ixx, color_image_t **Ixy, color_image_t **Iyy, color_image_t **Ixz, color_image_t **Iyz, const float delta_over3, const float gamma_over3, int f,
		float s) const {

	const v4sf dnorm = { datanorm, datanorm, datanorm, datanorm };
	const v4sf hdover3 = { delta_over3, delta_over3, delta_over3, delta_over3 };
	const v4sf hgover3 = { gamma_over3, gamma_over3, gamma_over3, gamma_over3 };

	v4sf factor = { s, s, s, s };
	v4sf factorsq = factor * factor;

	if (s == 0) {
		throw logic_error("Frame compared to reference frame is the reference frame itself!");
	}

	// since I_ref - I_s instead of I_s - I_ref
	if (s >= 0)
		factor = -factor;

	v4sf *dup = (v4sf*) du->data, *dvp = (v4sf*) dv->data, *maskp = (v4sf*) mask->data, *weight1p = (v4sf*) channel_w->c1, *weight2p = (v4sf*) channel_w->c2, *weight3p = (v4sf*) channel_w->c3, *a11p =
			(v4sf*) a11->data, *a12p = (v4sf*) a12->data, *a22p = (v4sf*) a22->data, *b1p = (v4sf*) b1->data, *b2p = (v4sf*) b2->data, *ix1p = (v4sf*) Ix[f]->c1, *iy1p = (v4sf*) Iy[f]->c1, *iz1p =
			(v4sf*) Iz[f]->c1, *ixx1p = (v4sf*) Ixx[f]->c1, *ixy1p = (v4sf*) Ixy[f]->c1, *iyy1p = (v4sf*) Iyy[f]->c1, *ixz1p = (v4sf*) Ixz[f]->c1, *iyz1p = (v4sf*) Iyz[f]->c1, *ix2p =
			(v4sf*) Ix[f]->c2, *iy2p = (v4sf*) Iy[f]->c2, *iz2p = (v4sf*) Iz[f]->c2, *ixx2p = (v4sf*) Ixx[f]->c2, *ixy2p = (v4sf*) Ixy[f]->c2, *iyy2p = (v4sf*) Iyy[f]->c2, *ixz2p = (v4sf*) Ixz[f]->c2,
			*iyz2p = (v4sf*) Iyz[f]->c2, *ix3p = (v4sf*) Ix[f]->c3, *iy3p = (v4sf*) Iy[f]->c3, *iz3p = (v4sf*) Iz[f]->c3, *ixx3p = (v4sf*) Ixx[f]->c3, *ixy3p = (v4sf*) Ixy[f]->c3, *iyy3p =
					(v4sf*) Iyy[f]->c3, *ixz3p = (v4sf*) Ixz[f]->c3, *iyz3p = (v4sf*) Iyz[f]->c3;

	v4sf tmp, tmp2, tmp3, tmp4, tmp5, tmp6, n1, n2, n3, n4, n5, n6;

	int i;
	for (i = 0; i < du->height * du->stride / 4; i++) {
		// dpsi color
		if (delta_over3) {

			tmp = (*weight1p) * (*iz1p + (*ix1p) * factor * (*dup) + (*iy1p) * factor * (*dvp));
			tmp2 = (*weight2p) * (*iz2p + (*ix2p) * factor * (*dup) + (*iy2p) * factor * (*dvp));
			tmp3 = (*weight3p) * (*iz3p + (*ix3p) * factor * (*dup) + (*iy3p) * factor * (*dvp));

			if (!dt_norm) {
				// without NORMALIZATION
				tmp = (*maskp) * hdover3 * robust_color->derivative(tmp * tmp / factorsq + tmp2 * tmp2 / factorsq + tmp3 * tmp3 / factorsq);
				tmp /= factorsq;

				tmp2 = tmp * (*weight1p) * factor;
				*b1p -= tmp2 * (*iz1p) * (*ix1p);
				*b2p -= tmp2 * (*iz1p) * (*iy1p);
				tmp2 = tmp2 * factor; // factorsq
				*a11p += tmp2 * (*ix1p) * (*ix1p);
				*a12p += tmp2 * (*ix1p) * (*iy1p);
				*a22p += tmp2 * (*iy1p) * (*iy1p);

				tmp2 = tmp * factor * (*weight2p);
				*b1p -= tmp2 * (*iz2p) * (*ix2p);
				*b2p -= tmp2 * (*iz2p) * (*iy2p);
				tmp2 = tmp2 * factor; // factorsq
				*a11p += tmp2 * (*ix2p) * (*ix2p);
				*a12p += tmp2 * (*ix2p) * (*iy2p);
				*a22p += tmp2 * (*iy2p) * (*iy2p);

				tmp2 = tmp * factor * (*weight3p);
				*b1p -= tmp2 * (*iz3p) * (*ix3p);
				*b2p -= tmp2 * (*iz3p) * (*iy3p);
				tmp2 = tmp * factor; // factorsq
				*a11p += tmp2 * (*ix3p) * (*ix3p);
				*a12p += tmp2 * (*ix3p) * (*iy3p);
				*a22p += tmp2 * (*iy3p) * (*iy3p);
			} else {
				// with NORMALIZATION
				n1 = factorsq * (*ix1p) * (*ix1p) + factorsq * (*iy1p) * (*iy1p) + dnorm;
				n2 = factorsq * (*ix2p) * (*ix2p) + factorsq * (*iy2p) * (*iy2p) + dnorm;
				n3 = factorsq * (*ix3p) * (*ix3p) + factorsq * (*iy3p) * (*iy3p) + dnorm;

				tmp = (*maskp) * hdover3 * robust_color->derivative(tmp * tmp / n1 + tmp2 * tmp2 / n2 + tmp3 * tmp3 / n3);
				tmp3 = tmp / n3;
				tmp2 = tmp / n2;
				tmp /= n1;

				tmp = tmp * (*weight1p) * factor;
				*b1p -= tmp * (*iz1p) * (*ix1p);
				*b2p -= tmp * (*iz1p) * (*iy1p);
				tmp = tmp * factor; // factorsq
				*a11p += tmp * (*ix1p) * (*ix1p);
				*a12p += tmp * (*ix1p) * (*iy1p);
				*a22p += tmp * (*iy1p) * (*iy1p);

				tmp2 = tmp2 * (*weight2p) * factor;
				*b1p -= tmp2 * (*iz2p) * (*ix2p);
				*b2p -= tmp2 * (*iz2p) * (*iy2p);
				tmp2 = tmp2 * factor; // factorsq
				*a11p += tmp2 * (*ix2p) * (*ix2p);
				*a12p += tmp2 * (*ix2p) * (*iy2p);
				*a22p += tmp2 * (*iy2p) * (*iy2p);

				tmp3 = tmp3 * (*weight3p) * factor;
				*b1p -= tmp3 * (*iz3p) * (*ix3p);
				*b2p -= tmp3 * (*iz3p) * (*iy3p);
				tmp3 = tmp3 * factor; // factorsq
				*a11p += tmp3 * (*ix3p) * (*ix3p);
				*a12p += tmp3 * (*ix3p) * (*iy3p);
				*a22p += tmp3 * (*iy3p) * (*iy3p);
			}
		}

		// dpfactori gradient
		tmp = (*weight1p) * (*ixz1p + (*ixx1p) * factor * (*dup) + (*ixy1p) * factor * (*dvp));
		tmp2 = (*weight1p) * (*iyz1p + (*ixy1p) * factor * (*dup) + (*iyy1p) * factor * (*dvp));
		tmp3 = (*weight2p) * (*ixz2p + (*ixx2p) * factor * (*dup) + (*ixy2p) * factor * (*dvp));
		tmp4 = (*weight2p) * (*iyz2p + (*ixy2p) * factor * (*dup) + (*iyy2p) * factor * (*dvp));
		tmp5 = (*weight3p) * (*ixz3p + (*ixx3p) * factor * (*dup) + (*ixy3p) * factor * (*dvp));
		tmp6 = (*weight3p) * (*iyz3p + (*ixy3p) * factor * (*dup) + (*iyy3p) * factor * (*dvp));

		if (!dt_norm) {
			// without NORMALIZATION
			tmp = (*maskp) * hgover3
					* robust_grad->derivative(tmp * tmp / factorsq + tmp2 * tmp2 / factorsq + tmp3 * tmp3 / factorsq + tmp4 * tmp4 / factorsq + tmp5 * tmp5 / factorsq + tmp6 * tmp6 / factorsq);
			tmp /= factorsq;

			tmp2 = tmp * (*weight1p) * factor;
			*b1p -= tmp2 * (*ixx1p) * (*ixz1p) + tmp2 * (*ixy1p) * (*iyz1p);
			*b2p -= tmp2 * (*iyy1p) * (*iyz1p) + tmp2 * (*ixy1p) * (*ixz1p);
			tmp2 = tmp2 * factor; // factorsq
			*a11p += tmp2 * factorsq * (*ixx1p) * (*ixx1p) + tmp2 * factorsq * (*ixy1p) * (*ixy1p);
			*a12p += tmp2 * factorsq * (*ixx1p) * (*ixy1p) + tmp2 * factorsq * (*ixy1p) * (*iyy1p);
			*a22p += tmp2 * factorsq * (*iyy1p) * (*iyy1p) + tmp2 * factorsq * (*ixy1p) * (*ixy1p);

			tmp2 = tmp * (*weight2p) * factor;
			*b1p -= tmp2 * (*ixx2p) * (*ixz2p) + tmp2 * (*ixy2p) * (*iyz2p);
			*b2p -= tmp2 * (*iyy2p) * (*iyz2p) + tmp2 * (*ixy2p) * (*ixz2p);
			tmp2 = tmp2 * factor; // factorsq
			*a11p += tmp2 * (*ixx2p) * (*ixx2p) + tmp2 * (*ixy2p) * (*ixy2p);
			*a12p += tmp2 * (*ixx2p) * (*ixy2p) + tmp2 * (*ixy2p) * (*iyy2p);
			*a22p += tmp2 * (*iyy2p) * (*iyy2p) + tmp2 * (*ixy2p) * (*ixy2p);

			tmp2 = tmp * (*weight3p) * factor;
			*b1p -= tmp2 * (*ixx3p) * (*ixz3p) + tmp2 * (*ixy3p) * (*iyz3p);
			*b2p -= tmp2 * (*iyy3p) * (*iyz3p) + tmp2 * (*ixy3p) * (*ixz3p);
			tmp2 = tmp2 * factor; // factorsq
			*a11p += tmp2 * (*ixx3p) * (*ixx3p) + tmp2 * (*ixy3p) * (*ixy3p);
			*a12p += tmp2 * (*ixx3p) * (*ixy3p) + tmp2 * (*ixy3p) * (*iyy3p);
			*a22p += tmp2 * (*iyy3p) * (*iyy3p) + tmp2 * (*ixy3p) * (*ixy3p);
		} else {
			// with NORMALIZATION
			n1 = factorsq * (*ixx1p) * (*ixx1p) + factorsq * (*ixy1p) * (*ixy1p) + dnorm;
			n2 = factorsq * (*iyy1p) * (*iyy1p) + factorsq * (*ixy1p) * (*ixy1p) + dnorm;
			n3 = factorsq * (*ixx2p) * (*ixx2p) + factorsq * (*ixy2p) * (*ixy2p) + dnorm;
			n4 = factorsq * (*iyy2p) * (*iyy2p) + factorsq * (*ixy2p) * (*ixy2p) + dnorm;
			n5 = factorsq * (*ixx3p) * (*ixx3p) + factorsq * (*ixy3p) * (*ixy3p) + dnorm;
			n6 = factorsq * (*iyy3p) * (*iyy3p) + factorsq * (*ixy3p) * (*ixy3p) + dnorm;

			tmp = (*maskp) * hgover3 * robust_grad->derivative(tmp * tmp / n1 + tmp2 * tmp2 / n2 + tmp3 * tmp3 / n3 + tmp4 * tmp4 / n4 + tmp5 * tmp5 / n5 + tmp6 * tmp6 / n6);
			tmp6 = tmp / n6;
			tmp5 = tmp / n5;
			tmp4 = tmp / n4;
			tmp3 = tmp / n3;
			tmp2 = tmp / n2;
			tmp /= n1;

			tmp = tmp * (*weight1p) * factor;
			tmp2 = tmp2 * (*weight1p) * factor;
			*b1p -= tmp * (*ixx1p) * (*ixz1p) + tmp2 * (*ixy1p) * (*iyz1p);
			*b2p -= tmp2 * (*iyy1p) * (*iyz1p) + tmp * (*ixy1p) * (*ixz1p);
			tmp = tmp * factor; // factorsq
			tmp2 = tmp2 * factor; // factorsq
			*a11p += tmp * (*ixx1p) * (*ixx1p) + tmp2 * (*ixy1p) * (*ixy1p);
			*a12p += tmp * (*ixx1p) * (*ixy1p) + tmp2 * (*ixy1p) * (*iyy1p);
			*a22p += tmp2 * (*iyy1p) * (*iyy1p) + tmp * (*ixy1p) * (*ixy1p);

			tmp3 = tmp3 * (*weight2p) * factor;
			tmp4 = tmp4 * (*weight2p) * factor;
			*b1p -= tmp3 * (*ixx2p) * (*ixz2p) + tmp4 * (*ixy2p) * (*iyz2p);
			*b2p -= tmp4 * (*iyy2p) * (*iyz2p) + tmp3 * (*ixy2p) * (*ixz2p);
			tmp3 = tmp3 * factor; // factorsq
			tmp4 = tmp4 * factor; // factorsq
			*a11p += tmp3 * (*ixx2p) * (*ixx2p) + tmp4 * (*ixy2p) * (*ixy2p);
			*a12p += tmp3 * (*ixx2p) * (*ixy2p) + tmp4 * (*ixy2p) * (*iyy2p);
			*a22p += tmp4 * (*iyy2p) * (*iyy2p) + tmp3 * (*ixy2p) * (*ixy2p);

			tmp5 = tmp5 * (*weight3p) * factor;
			tmp6 = tmp6 * (*weight3p) * factor;
			*b1p -= tmp5 * (*ixx3p) * (*ixz3p) + tmp6 * (*ixy3p) * (*iyz3p);
			*b2p -= tmp6 * (*iyy3p) * (*iyz3p) + tmp5 * (*ixy3p) * (*ixz3p);
			tmp5 = tmp5 * factor; // factorsq
			tmp6 = tmp6 * factor; // factorsq
			*a11p += tmp5 * (*ixx3p) * (*ixx3p) + tmp6 * (*ixy3p) * (*ixy3p);
			*a12p += tmp5 * (*ixx3p) * (*ixy3p) + tmp6 * (*ixy3p) * (*iyy3p);
			*a22p += tmp6 * (*iyy3p) * (*iyy3p) + tmp5 * (*ixy3p) * (*ixy3p);
		}

		dup += 1;
		dvp += 1;
		maskp += 1;
		weight1p += 1;
		weight2p += 1;
		weight3p += 1;
		a11p += 1;
		a12p += 1;
		a22p += 1;
		b1p += 1;
		b2p += 1;

		ix1p += 1;
		iy1p += 1;
		iz1p += 1;
		ixx1p += 1;
		ixy1p += 1;
		iyy1p += 1;
		ixz1p += 1;
		iyz1p += 1;

		ix2p += 1;
		iy2p += 1;
		iz2p += 1;
		ixx2p += 1;
		ixy2p += 1;
		iyy2p += 1;
		ixz2p += 1;
		iyz2p += 1;

		ix3p += 1;
		iy3p += 1;
		iz3p += 1;
		ixx3p += 1;
		ixy3p += 1;
		iyy3p += 1;
		ixz3p += 1;
		iyz3p += 1;
	}
}

/* compute local smoothness weight as a sigmoid on image gradient*/
image_t* Variational_AUX_MT::compute_dpsis_weight(const color_image_t *im, float coef, const convolution_t *deriv, float mn_1, float mn_2, float mn_3, float mx_1, float mx_2, float mx_3) {
	image_t* lum = image_new(im->width, im->height), *lum_x = image_new(im->width, im->height), *lum_y = image_new(im->width, im->height);
	int i;

	// compute luminance
	v4sf *im1p = (v4sf*) im->c1, *im2p = (v4sf*) im->c2, *im3p = (v4sf*) im->c3, *lump = (v4sf*) lum->data;
	for (i = 0; i < im->height * im->stride / 4; i++) {
		*lump = (0.299f * ((*im1p) - mn_1) / (mx_1 - mn_1) + 0.587f * ((*im2p) - mn_2) / (mx_2 - mn_2) + 0.114f * ((*im3p)) - mn_3) / (mx_3 - mn_3); // channels normalized to 0..1
		lump += 1;
		im1p += 1;
		im2p += 1;
		im3p += 1;
	}

	// compute derivatives with five-point tencil
	convolve_horiz(lum_x, lum, deriv);
	convolve_vert(lum_y, lum, deriv);

	// compute lum norm
	lump = (v4sf*) lum->data;
	v4sf *lumxp = (v4sf*) lum_x->data, *lumyp = (v4sf*) lum_y->data;
	for (i = 0; i < lum->height * lum->stride / 4; i++) {
		*lump = -coef * __builtin_ia32_sqrtps((*lumxp) * (*lumxp) + (*lumyp) * (*lumyp));
		lump[0][0] = expf((float) lump[0][0]);
		lump[0][1] = expf((float) lump[0][1]);
		lump[0][2] = expf((float) lump[0][2]);
		lump[0][3] = expf((float) lump[0][3]);
		lump += 1;
		lumxp += 1;
		lumyp += 1;
	}
	image_delete(lum_x);
	image_delete(lum_y);
	return lum;
}

void Variational_AUX_MT::compute_dpsis_weight(const color_image_t *im, image_t* lum, image_t* lum_x, image_t* lum_y, float coef, const convolution_t *deriv, float avg_1, float avg_2, float avg_3,
		float std_1, float std_2, float std_3, bool hbit) {
	int i;

	// compute luminance
	v4sf *im1p = (v4sf*) im->c1, *im2p = (v4sf*) im->c2, *im3p = (v4sf*) im->c3, *lump = (v4sf*) lum->data;
	for (i = 0; i < im->height * im->stride / 4; i++) {
		if (hbit)
			*lump = (0.299f * ((*im1p) * std_1 + avg_1) + 0.587f * ((*im2p) * std_2 + avg_2) + 0.114f * ((*im3p) * std_3 + avg_3)) / 65535.0f; // channels normalized to 0..1
		else
			*lump = (0.299f * ((*im1p) * std_1 + avg_1) + 0.587f * ((*im2p) * std_2 + avg_2) + 0.114f * ((*im3p) * std_3 + avg_3)) / 255.0f; // channels normalized to 0..1

		lump += 1;
		im1p += 1;
		im2p += 1;
		im3p += 1;
	}

	// compute derivatives with five-point tencil
	convolve_horiz(lum_x, lum, deriv);
	convolve_vert(lum_y, lum, deriv);

	// compute lum norm
	lump = (v4sf*) lum->data;
	v4sf *lumxp = (v4sf*) lum_x->data, *lumyp = (v4sf*) lum_y->data;
	for (i = 0; i < lum->height * lum->stride / 4; i++) {
		*lump = -coef * __builtin_ia32_sqrtps((*lumxp) * (*lumxp) + (*lumyp) * (*lumyp));
		lump[0][0] = 0.5f * expf((float) lump[0][0]);
		lump[0][1] = 0.5f * expf((float) lump[0][1]);
		lump[0][2] = 0.5f * expf((float) lump[0][2]);
		lump[0][3] = 0.5f * expf((float) lump[0][3]);

		lumxp[0][0] = 0.5f * expf(-coef * fabs((float) lumxp[0][0]));
		lumxp[0][1] = 0.5f * expf(-coef * fabs((float) lumxp[0][1]));
		lumxp[0][2] = 0.5f * expf(-coef * fabs((float) lumxp[0][2]));
		lumxp[0][3] = 0.5f * expf(-coef * fabs((float) lumxp[0][3]));

		lumyp[0][0] = 0.5f * expf(-coef * fabs((float) lumyp[0][0]));
		lumyp[0][1] = 0.5f * expf(-coef * fabs((float) lumyp[0][1]));
		lumyp[0][2] = 0.5f * expf(-coef * fabs((float) lumyp[0][2]));
		lumyp[0][3] = 0.5f * expf(-coef * fabs((float) lumyp[0][3]));

		lump += 1;
		lumxp += 1;
		lumyp += 1;
	}
}

/* warp a color image according to a flow. src is the input image, wx and wy, the input flow. dst is the warped image and mask contains 0 or 1 if the pixels goes outside/inside image boundaries */
void Variational_AUX_MT::image_warp(color_image_t *dst, image_t *mask, const color_image_t *src, const image_t *wx, const image_t *wy, int factor) {
	if (factor == 0) {
		memcpy(dst->c1, src->c1, 3 * src->stride * src->height * sizeof(float));
		if (mask != NULL)
			memset(mask->data, 1, src->stride * src->height * sizeof(float));
		return;
	}

	int i, j, offset, x, y, x1, x2, y1, y2;
	float xx, yy, dx, dy;
	for (j = 0; j < src->height; j++) {
		offset = j * src->stride;
		for (i = 0; i < src->width; i++, offset++) {
			xx = i + factor * wx->data[offset];
			yy = j + factor * wy->data[offset];
			x = floor(xx);
			y = floor(yy);
			dx = xx - x;
			dy = yy - y;
			if (mask != NULL) {
				mask->data[offset] = (xx >= 0 && xx <= src->width - 1 && yy >= 0 && yy <= src->height - 1);
			}
			x1 = RECTIFY(x, src->width);
			x2 = RECTIFY(x + 1, src->width);
			y1 = RECTIFY(y, src->height);
			y2 = RECTIFY(y + 1, src->height);
			dst->c1[offset] = src->c1[y1 * src->stride + x1] * (1.0f - dx) * (1.0f - dy) + src->c1[y1 * src->stride + x2] * dx * (1.0f - dy) + src->c1[y2 * src->stride + x1] * (1.0f - dx) * dy
					+ src->c1[y2 * src->stride + x2] * dx * dy;
			dst->c2[offset] = src->c2[y1 * src->stride + x1] * (1.0f - dx) * (1.0f - dy) + src->c2[y1 * src->stride + x2] * dx * (1.0f - dy) + src->c2[y2 * src->stride + x1] * (1.0f - dx) * dy
					+ src->c2[y2 * src->stride + x2] * dx * dy;
			dst->c3[offset] = src->c3[y1 * src->stride + x1] * (1.0f - dx) * (1.0f - dy) + src->c3[y1 * src->stride + x2] * dx * (1.0f - dy) + src->c3[y2 * src->stride + x1] * (1.0f - dx) * dy
					+ src->c3[y2 * src->stride + x2] * dx * dy;
		}
	}
}

void Variational_AUX_MT::optimizeOcc(image_t* occlusions, image_t **mask, color_image_t **Iz, color_image_t **Iz_to_ref, color_image_t **Ixz, color_image_t **Iyz, color_image_t **Ixz_to_ref,
		color_image_t **Iyz_to_ref, int ref, const vector<float> rho, const vector<float> omega, float delta_over3, float gamma_over3, float penalty, float alpha, int graphc_it) {

	int height = occlusions->height;
	int width = occlusions->width;
	int stride = occlusions->stride;

	uint32_t num_labels = 2;

	const v4sf hdover3 = { delta_over3, delta_over3, delta_over3, delta_over3 };
	const v4sf hgover3 = { gamma_over3, gamma_over3, gamma_over3, gamma_over3 };

	try {
		// variables: 	occlusions
		// occlusions: 	in the past = 0, in the future = 1		 	(prefer occlusions in the past)

		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width, height, num_labels);

		v4sf *maskp, *iz1p, *iz2p, *iz3p, *ixz1p, *ixz2p, *ixz3p, *iyz1p, *iyz2p, *iyz3p, *iz1p_ref, *iz2p_ref, *iz3p_ref, *ixz1p_ref, *ixz2p_ref, *ixz3p_ref, *iyz1p_ref, *iyz2p_ref, *iyz3p_ref;

		vector<v4sf> rho_v4(ref), omega_v4(ref);
		for (int s = 0; s < ref; s++) {
			rho_v4[s] = _mm_set1_ps(rho[s]);
			omega_v4[s] = _mm_set1_ps(omega[s]);
		}

		// first set up data costs individually
		// SSE
		for (int i = 0; i < height * stride / 4; i++) {
			vector<v4sf> energies(num_labels, zero);
			vector<v4sf> norm(num_labels, zero);

			// compute all data terms
			for (int s = 0; s < 2 * ref; s++) {
				maskp = (v4sf*) &mask[s]->data[i * 4];
				iz1p = (v4sf*) &Iz[s]->c1[i * 4];
				iz2p = (v4sf*) &Iz[s]->c2[i * 4];
				iz3p = (v4sf*) &Iz[s]->c3[i * 4];
				ixz1p = (v4sf*) &Ixz[s]->c1[i * 4];
				ixz2p = (v4sf*) &Ixz[s]->c2[i * 4];
				ixz3p = (v4sf*) &Ixz[s]->c3[i * 4];
				iyz1p = (v4sf*) &Iyz[s]->c1[i * 4];
				iyz2p = (v4sf*) &Iyz[s]->c2[i * 4];
				iyz3p = (v4sf*) &Iyz[s]->c3[i * 4];
				iz1p_ref = (v4sf*) &Iz_to_ref[s]->c1[i * 4];
				iz2p_ref = (v4sf*) &Iz_to_ref[s]->c2[i * 4];
				iz3p_ref = (v4sf*) &Iz_to_ref[s]->c3[i * 4];
				ixz1p_ref = (v4sf*) &Ixz_to_ref[s]->c1[i * 4];
				ixz2p_ref = (v4sf*) &Ixz_to_ref[s]->c2[i * 4];
				ixz3p_ref = (v4sf*) &Ixz_to_ref[s]->c3[i * 4];
				iyz1p_ref = (v4sf*) &Iyz_to_ref[s]->c1[i * 4];
				iyz2p_ref = (v4sf*) &Iyz_to_ref[s]->c2[i * 4];
				iyz3p_ref = (v4sf*) &Iyz_to_ref[s]->c3[i * 4];

				int idx = max(ref - s - 1, s - ref);

				// successive data term
				v4sf term = rho_v4[idx] * hdover3 * (*maskp) * robust_color->apply(((*iz1p) * (*iz1p) + (*iz2p) * (*iz2p) + (*iz3p) * (*iz3p)));
				term += rho_v4[idx] * hgover3 * (*maskp)
						* robust_grad->apply(((*ixz1p) * (*ixz1p) + (*ixz2p) * (*ixz2p) + (*ixz3p) * (*ixz3p) + (*iyz1p) * (*iyz1p) + (*iyz2p) * (*iyz2p) + (*iyz3p) * (*iyz3p)));

				// reference data term
				term += omega_v4[idx] * hdover3 * (*maskp) * robust_color->apply(((*iz1p_ref) * (*iz1p_ref) + (*iz2p_ref) * (*iz2p_ref) + (*iz3p_ref) * (*iz3p_ref)));
				term += omega_v4[idx] * hgover3 * (*maskp)
						* robust_grad->apply(
								((*ixz1p_ref) * (*ixz1p_ref) + (*ixz2p_ref) * (*ixz2p_ref) + (*ixz3p_ref) * (*ixz3p_ref) + (*iyz1p_ref) * (*iyz1p_ref) + (*iyz2p_ref) * (*iyz2p_ref)
										+ (*iyz3p_ref) * (*iyz3p_ref)));
				// sum energies
				if (s >= ref) {
					// occlusion in the past!
					energies[0] += term;
					norm[0] += (*maskp) * (rho_v4[idx] + rho_v4[idx] + omega_v4[idx] + omega_v4[idx]);
				} else {
					// occlusion in the future!
					energies[1] += term;
					norm[1] += (*maskp) * (rho_v4[idx] + rho_v4[idx] + omega_v4[idx] + omega_v4[idx]);
				}
			}

			for (uint32_t l = 0; l < num_labels; l++) {
				int sse_pix = i * 4;
				int x = sse_pix % stride;
				int y = sse_pix / stride;

				// avoid division by zero
				if (norm[l][0] == 0) norm[l][0] = 1;
				if (norm[l][1] == 0) norm[l][1] = 1;
				if (norm[l][2] == 0) norm[l][2] = 1;
				if (norm[l][3] == 0) norm[l][3] = 1;

				v4sf e = dt_scale_graphc * energies[l] / norm[l] + penalty * l; // prefer occlusions in the past!

				if ((x) < width)
					gc->setDataCost(x + y * width, l, (float) e[0]);
				if ((x + 1) < width)
					gc->setDataCost(x + y * width + 1, l, (float) e[1]);
				if ((x + 2) < width)
					gc->setDataCost(x + y * width + 2, l, (float) e[2]);
				if ((x + 3) < width)
					gc->setDataCost(x + y * width + 3, l, (float) e[3]);
			}
		}

		// next set up smoothness costs individually
		for (uint32_t l1 = 0; l1 < num_labels; l1++)
			for (uint32_t l2 = 0; l2 < num_labels; l2++) {
				float cost = 0;

				if (l1 != l2)
					cost += alpha;

				gc->setSmoothCost(l1, l2, cost);
			}

		gc->expansion(graphc_it); // run expansion for graphc_it iterations

		for (int y = 0; y < occlusions->height; y++) {
			int offset = y * occlusions->stride;
			for (int x = 0; x < occlusions->width; x++, offset++) {
				int l = gc->whatLabel(x + y * width);

				occlusions->data[offset] = 2 * l - 1; // occlusion in the past -1 and in the future 1
			}
		}

		delete gc;
	} catch (GCException e) {
		e.Report();
	}
}

void Variational_AUX_MT::select_robust_function(int who, int fct, float eps, float trunc) {
	switch (who) {
	case Robust_Grad:
		select_robust_function(robust_grad, fct, eps, trunc);
		break;
	case Robust_Reg:
		select_robust_function(robust_reg, fct, eps, trunc);
		break;
	default:
		select_robust_function(robust_color, fct, eps, trunc);
		break;
	}
}

void Variational_AUX_MT::select_robust_function(PenaltyFunction*& robust, int fct, float eps, float trunc) {
	if (robust != NULL) {
		delete robust;
		robust = NULL;
	}

	switch (fct) {
	case 0:
		robust = new QuadraticFunction();
		break;
	case 2:
		robust = new Lorentzian(eps);
		break;
	case 3:
		robust = new TruncModifiedL1Norm(eps, trunc);
		break;
	case 4:
		robust = new GemanMcClure(eps);
		break;
	default:
		robust = new ModifiedL1Norm(eps);
		break;
	}
}

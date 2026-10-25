# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Worker/frontend image for the DeepSeek-R1 Pareto-sweep recipes
# (inference/a4xmax/disaggregated-serving/dynamo/pareto-sweep).
#
# Reproduces the software stack the InferenceX GB300 recipes pin: an SGLang
# release image plus a released ai-dynamo wheel. Build for linux/arm64 (Grace).
#
#   DeepSeek-R1 NVFP4:  BASE_IMAGE=lmsysorg/sglang:v0.5.8.post1-cu130-runtime  DYNAMO_VERSION=0.8.1
#   DeepSeek-R1 FP8:    BASE_IMAGE=lmsysorg/sglang:v0.5.8.post1-cu130          DYNAMO_VERSION=0.8.0
#
#   docker buildx build --platform linux/arm64 \
#     --build-arg BASE_IMAGE=lmsysorg/sglang:v0.5.8.post1-cu130-runtime \
#     --build-arg DYNAMO_VERSION=0.8.1 \
#     -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/dynamo-sglang-dsr1:0.8.1 --push .
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.8.post1-cu130-runtime
FROM ${BASE_IMAGE}

ARG DYNAMO_VERSION=0.8.1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# NIXL ships its shared libraries inside the Python package; expose them at the
# path the Dynamo/UCX plugins expect.
RUN mkdir -p /opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu \
 && ln -sf /usr/local/lib/python3.*/dist-packages/.nixl_cu13.mesonpy.libs/*.so* \
      /opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu/ 2>/dev/null || true

RUN pip install --no-cache-dir \
      "ai-dynamo-runtime==${DYNAMO_VERSION}" \
      "ai-dynamo==${DYNAMO_VERSION}" \
      hf_transfer

# Sanity check: the Dynamo SGLang worker and frontend modules import.
RUN python3 -c "import dynamo.sglang, dynamo.frontend; print('dynamo', '${DYNAMO_VERSION}', 'ok')"

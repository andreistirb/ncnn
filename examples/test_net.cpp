// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "platform.h"
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN
#include <iostream>

using namespace std;
static int detect_net(std::vector<float>& cls_scores)
{
    ncnn::Net test_net;

    test_net.load_param("test_net_new.param");
    cout << "Loaded params" << endl;
    test_net.load_model("test_net_new.bin");

    ncnn::Mat in = ncnn::Mat(224, 224, 3);

    in.fill(1);
    const float mean_vals[3] = {0.3f, 0.4f, 0.2f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = test_net.create_extractor();

    ex.input("Placeholder:0", in);

    ncnn::Mat out;
    ncnn::Mat landmarks;
    ncnn::Mat visibility;
    ex.extract("prediction/classes/Softmax:0", out);
    ex.extract("prediction/landmarks/Squeeze:0", landmarks);
    ex.extract("prediction/landmarks/visibility/Softmax/0", visibility);
    printf("Landmarks channels: %d\n", landmarks.c);
    printf("Landmarks width: %d\n", landmarks.w);
    printf("Landmarks height: %d\n", landmarks.h);
    for (int i=0; i<landmarks.c*landmarks.w*landmarks.h; i++)
	    printf("Landmarks[%d] value is: %f\n",i,landmarks[i]);

    printf("Classes output width: %d\n",out.w);
    printf("Classes output height: %d\n", out.h);
    printf("Classes output channels: %d\n", out.c);
    for (int i=0; i<out.c*out.h*out.w; i++)
	    printf("Classes[%d] value is: %f\n",i,out[i]);

    printf("Visibility output width: %d\n", visibility.w);
    printf("visibility output height: %d\n", visibility.h);
    printf("visibility output channels: %d\n", visibility.c);
    for (int i=0; i<visibility.c*visibility.h*visibility.w; i++)
	    printf("Visibility[%d] value is: %f\n",i,visibility[i]);
    
    std::vector<ncnn::Blob> blobs = test_net.blobs;
    //printf("%s\n",blobs[0].name);
    //for(int i=0;i<blobs.size();i++){
    //	std::cout << "Layer " << blobs[blobs[i].producer].name << " produces " << blobs[i].name << " and is needed as input by: " << std::endl;
//	for (int j=0;j<blobs[i].consumers.size();j++)
//		std::cout << "\t\t " << blobs[blobs[i].consumers[j]].name << std::endl;
 //   }








    //cls_scores.resize(out.w);
    //for (int j=0; j<out.w; j++)
    //{
    //    cls_scores[j] = out[j];
    //}

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{

    std::vector<float> cls_scores;
    detect_net(cls_scores);

    //print_topk(cls_scores, 3);

    return 0;
}

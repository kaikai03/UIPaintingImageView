//
//  UIPaintingImageView.h
//  PaintView
//
//  Created by kk on 2018/11/18.
//  Copyright © 2018年 kk. All rights reserved.
//


#import "UIPaintingImageView.h"


using namespace cv;


@implementation UIPaintingImageView
@synthesize delegate = _delegate;
@synthesize pixelNum = _pixelNum;
@synthesize pRgbImageBuf = _pRgbImageBuf;
@synthesize imageWidth = _imageWidth;
@synthesize imageHeight = _imageHeight;
@synthesize bytesPerRow = _bytesPerRow;


/**
 kk_cannyKenel
 取得原图边界。
 
 @param cvGrayMat 原图的灰度图。
 @return edges 原图边界图。
 */
static cv::Mat kk_cannyKenel(cv::Mat cvGrayMat){
    cv::Mat edges;
    cv::Canny(cvGrayMat, edges, 0, 60,3,true);
    return edges;
}


/**
 kk_sortContoursIndexsCorrelativeToFace
 区块根据脸位置进行排序。
 
 @param pVec 区块序列信息，具体内容为：Vec4i(faceIndex,contourIndex,hierarchy[contourIndex][0],int(isFaceCenter))。
 @return 。
 */
+ (void)kk_sortContoursIndexsCorrelativeToFace:(std::vector<cv::Vec4i>*)pVec{
    std::vector<cv::Vec4i> &vec = *pVec;
    for(int i=0; i<vec.size(); ++i){
        cv::Vec4i v = vec[i];
        if (v[3]==1){
            int face = v[0];
//            int contour = v[1];
//            int next = v[2];
            bool hadExchanged = false;
            for(int j=0; j<vec.size(); ++j){
                if(vec[j][0]==face){
                    if (!hadExchanged) {
                        //isCenter==1的换到同一个脸的第一个，
                        cv::Vec4i tmp = vec[i];
                        vec[i] = vec[j];
                        vec[j] = tmp;
                        //通常来说，一个脸的数据是在“一起”的，也就说如果需要换，只需要换到“一块”的第一个就好。
                        hadExchanged = true;
                        continue;
                    }
                    //此处未来可用于区块临近关系排序
                }
                if(vec[j][0]>face)break;
            }
        }
    }
}


/**
 splitByKmeans
 通过kmeans进行分区，本函数暂时废弃。
 
 @param mat 原图像的mat。
 @return markers 分区后图像。
 */
static cv::Mat splitByKmeans(cv::Mat *mat){
    cv::Mat cvMat = *mat;
    int sampleCount = cvMat.cols*cvMat.rows;//所有的像素
    int clusterCount = 5;//分类数
    Mat points = Mat(sampleCount, 1, CV_32FC3);
    Mat labels;//聚类后的标签
    Mat center(clusterCount, 1, points.type());//聚类后的类别的中心

    int index;
    for (int i = 0; i < cvMat.rows; i++)
    {
        for (int j = 0; j < cvMat.cols; j++)
        {
            index = i*cvMat.cols + j;
            Vec3b bgr = cvMat.at<Vec3b>(i, j);
            //将图像中的每个通道的数据分别赋值给points的值
            points.at<Vec3f>(index, 0)[0] = float(bgr[0]);
            points.at<Vec3f>(index, 0)[1] = float(bgr[1]);
            points.at<Vec3f>(index, 0)[2] = float(bgr[2]);
        }
    }

    TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT,10,1.0);
    kmeans(points, clusterCount, labels, criteria, 3, KMEANS_PP_CENTERS, center);

    cv::Mat markers(cvMat.size(),CV_8S,cv::Scalar(0));
    for (int i = 0; i < cvMat.rows; i++)
    {
        for (int j = 0; j < cvMat.cols; j++)
        {
            int index = i*cvMat.cols + j;
            int label = labels.at<int>(index);//每一个像素属于哪个标签
            label = (label)*(255/(clusterCount-1));
            markers.at<uchar>(i, j) = uchar(label);//对结果图中的每一个通道进行赋值
        }
    }
    return markers;
}


/**
 kk_cutArea
 对原图划分区域，排序，输出带绘画顺序的不规则多边形区块。
 区块以面积大小及相关程度进行排序。
 例如，最先找到最大的区块，然后是与最大区块相邻的区块，再然后是第二大的区块，以此类推。
 当faces不为空时，通过面部区域与区块匹配，进行绘制顺序加权。
 
 @param cvGrayMat 原图像的灰度图mat。
 @param originMat 原图像的mat。
 @param faces 面部坐标。
 @param imageWidth 原图高度。
 @param imageHeight 原图宽度。
 @return approxedContours 图像分区后的区块序列，其序列顺序为后续图像绘制顺序。
 */
static std::vector<std::vector<cv::Point>> kk_cutArea(cv::Mat cvGrayMat, cv::Mat originMat ,std::vector<cv::Rect> faces, int imageWidth,int imageHeight){
    
    cv::Mat c2v,opening,backgroud;


    CvScalar calar = cv::mean(cvGrayMat);
    cv::threshold(cvGrayMat, c2v, calar.val[0], 255, cv::THRESH_BINARY);//cv::THRESH_OTSU + cv::THRESH_BINARY_INV
    //开侵蚀
    cv::erode(c2v,opening,cv::Mat(),cv::Point(-1,-1),2);
    //膨胀
    cv::dilate(c2v, backgroud,  cv::getStructuringElement(2, cv::Size(3,3)), cv::Point(-1,-1), 3);
    cv::threshold(backgroud, backgroud, 1, 255, cv::THRESH_BINARY_INV);


    cv::Mat markers(c2v.size(),CV_8S,cv::Scalar(0));
    //创建标记图像
    markers = opening + backgroud;
    markers.convertTo(markers,CV_32S);
    cv::connectedComponents(opening,markers, 8, CV_32S);
    cv::watershed(originMat, markers);
    
    

    //分块包围
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(markers, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);


    std::vector<std::vector<cv::Point>> facesVec;
    for(cv::Rect faceRect:faces){
        std::vector<cv::Point> faceVec;
        faceVec.push_back(cv::Point(faceRect.x,faceRect.y));
        faceVec.push_back(cv::Point(faceRect.x+faceRect.width,faceRect.y));
        faceVec.push_back(cv::Point(faceRect.x+faceRect.width,faceRect.y+faceRect.height));
        faceVec.push_back(cv::Point(faceRect.x,faceRect.y+faceRect.height));
        facesVec.push_back(faceVec);
    }

    //存放脸与分块交集的4维信息向量:faceNumber[0:],contourIndex[-1:],NextContourIndex[-1:],isFaceCenter[0,1]
    std::vector<cv::Vec4i> ContoursIndexsCorrelativeToFace;
    //将分块转换为多边形坐标，最后会以绘画顺序相关的顺序返回
    //tip：多边形和原分块，共享index和hierarchy信息，
    std::vector<std::vector<cv::Point>> approxedContours;
    
    //有脸条件下，多边形包边，以及判断脸区域。结果记录与ContoursIndexsCorrelativeToFace
    for(int faceIndex=0;faceIndex<faces.size();++faceIndex)
        for(int contourIndex=0;contourIndex<contours.size();++contourIndex){
            //分水岭分块区域转为多边形，并且计算各多边形与脸的关系，将与脸有关的部分提前。
            std::vector<cv::Point> approx;
            bool isFace = false;
            bool isFaceCenter = false;
            
            if (faceIndex==0) {
                cv::approxPolyDP(contours[contourIndex], approx, 5, true);
                approxedContours.push_back(approx);
            }else{
                approx = approxedContours[contourIndex];
            }
            
            //脸与分块的是否有关系
            for(int appPointIndex=0;appPointIndex<approx.size();++appPointIndex){
                isFace = cv::pointPolygonTest(facesVec[faceIndex], approx[appPointIndex], false)>=0?true:false;
                if (isFace){
                    break;
                }
            }
            
            //与脸有关的多边形会有很多，但是和脸的中心点有关的，一般情况下只有一个区，除非多边形包边出错了。
            //做一个中心标记，用于绘图优先度的排序
            isFaceCenter = cv::pointPolygonTest(approx, cv::Point(faces[faceIndex].x+faces[faceIndex].width/2,faces[faceIndex].y+faces[faceIndex].height/2), false)>=0?true:false;
            if (isFace) {
                ContoursIndexsCorrelativeToFace.push_back(Vec4i(faceIndex,contourIndex,hierarchy[contourIndex][0],int(isFaceCenter)));
            }
        }
    
    
    if (ContoursIndexsCorrelativeToFace.size()>0) {
        //排序，将脸中心提前至一个脸相关块的最前面。
        [UIPaintingImageView kk_sortContoursIndexsCorrelativeToFace:&ContoursIndexsCorrelativeToFace];
        
        //有脸时区块排序：脸的中心，然后是其临近当前脸相关区域，然后是其他脸，如此循环
        vector<int> hadExchangeIndexVec;
        for(int i=0; i<ContoursIndexsCorrelativeToFace.size(); ++i){
            cv::Vec4i v = ContoursIndexsCorrelativeToFace[i];
            int contour = v[1];
            //备用int face = v[0],next = v[2],isCenter = v[3];
            //判断是否之前就被移动过，意思是分处于两个脸，但是又交叉区块，如果已经处理过，则跳过
            vector<int>::iterator it;
            it=find(hadExchangeIndexVec.begin(),hadExchangeIndexVec.end(),contour);
            if (it!=hadExchangeIndexVec.end())continue;
            //根据与脸的关系进行换位
            std::vector<cv::Point> tmp = approxedContours[i];
            approxedContours[i] = approxedContours[contour];
            approxedContours[contour] = tmp;
            //记录移动过的index
            hadExchangeIndexVec.push_back(contour);
        }
    }else{
        ////无脸时区块生成与，优先中心，连续找出两双边10%个，然后是最大区域，
        cv::Point centerPoint = cv::Point(imageWidth/2,imageHeight/2);
        std::vector<cv::Point> approx;
        int centerContourIndex = -1;
        int largestAreaIndex = -1;
        double largestSize = -1;
        for(int contourIndex=0; contourIndex<contours.size(); ++contourIndex){
            //由于有脸时，会进行多边形与脸匹配，为了减少循环次数，故与生成多边形的逻辑合并。
            //所以在无脸部分，需要另外生成多边形。
            cv::approxPolyDP(contours[contourIndex], approx, 5, true);
            approxedContours.push_back(approx);
            
            //计算每个多边形面积
            double curArea = contourArea(contours[contourIndex]);
            if (curArea>largestSize) {
                largestSize = curArea;
                largestAreaIndex = contourIndex;
            }
            
            if(centerContourIndex <= 0){
                //找到中心点在区域内则该逻辑不再进入
                centerContourIndex = cv::pointPolygonTest(approx, centerPoint, false)>=0?contourIndex:-1;
            }
//            NSLog(@"%d:%d,%d,%d,%d",contourIndex,hierarchy[contourIndex][0],hierarchy[contourIndex][1],hierarchy[contourIndex][2],hierarchy[contourIndex][3]);
        }
//        printf("largestAreaIndex:%d:%f \n",largestAreaIndex, largestSize);
//        printf("centerContourIndex:%d \n",centerContourIndex);
        
        //开始排序
        int sortIndex=0;
        vector<int> hadExchangeIndexVec;
        if (centerContourIndex != -1) {//如果存在中心多边形，则由中心向前后两个方向跟去。
            std::vector<cv::Point> tmp = approxedContours[0];
            approxedContours[0] = approxedContours[centerContourIndex];
            approxedContours[centerContourIndex] = tmp;
            int next = hierarchy[centerContourIndex][0];
            int previour = hierarchy[centerContourIndex][1];
            sortIndex = 1;
            hadExchangeIndexVec.push_back(centerContourIndex);
            
            for(; sortIndex<int(contours.size()*0.1); ){
                if (next!=-1) {
                    std::vector<cv::Point> tmp = approxedContours[sortIndex];
                    approxedContours[sortIndex] = approxedContours[next];
                    approxedContours[next] = tmp;
                    next = hierarchy[next][0];
                    
                    sortIndex+=1;
                    hadExchangeIndexVec.push_back(next);
                }
                if (previour!=-1) {
                    std::vector<cv::Point> tmp = approxedContours[sortIndex];
                    approxedContours[sortIndex] = approxedContours[previour];
                    approxedContours[previour] = tmp;
                    previour = hierarchy[previour][1];
                    
                    sortIndex+=1;
                    hadExchangeIndexVec.push_back(previour);
                }
                if (previour<=0 && next<=0) {
                    break;
                }
//                printf("sortIndex:%d,next:%d,previour:%d \n",sortIndex,next,previour);
            }
            
            //开始处理最大面积的区域
            if (largestAreaIndex != -1) {
                vector<int>::iterator it;
                it=find(hadExchangeIndexVec.begin(),hadExchangeIndexVec.end(),largestAreaIndex);
                if (it!=hadExchangeIndexVec.end()){
                    //最大面积区域已包含在中心区域，不做任何工作
                    printf("最大面积区域已包含在中心区域，不做任何工作 \n");
                }else{
                    //最大面积区域未包含在中心区域，向上提前至最开始未被移动的index
                    std::vector<cv::Point> tmp = approxedContours[sortIndex];
                    approxedContours[sortIndex] = approxedContours[largestAreaIndex];
                    approxedContours[largestAreaIndex] = tmp;
                    
                    sortIndex+=1;//本质多余，但是怕万一后面继续加逻辑忘记这个地方。
                }
            }
            //////最大面积的区域处理完毕
        }
        //////排序结束
    }
    ////////无脸时区块生成与排序，完全结束
    
    //测试绘图
//    for(int index=0;index<contours.size();++index){
//        vector<cv::Point> approx;
//        std::vector<std::vector<cv::Point>> ttt;
//        ttt.push_back(approxedContours[index]);
//        cv::drawContours(originMat, ttt, -1, cv::Scalar(arc4random()%250,arc4random()%250,arc4random()%250),3);
//    }
//    NSLog(@"contours.size,%lu",contours.size());
    //
    
    return approxedContours;
}


/**
 kk_detectFaceByhaar
 识别面部区域。用于后续区块排序加权。存在面部的区域将被优先绘制。
 
 
 @param cvGrayMat 原图像的灰度图mat。
 @return faces 识别的面部序列，内容为面部区域的rect坐标。
 */
static std::vector<cv::Rect> kk_detectFaceByhaar(cv::Mat cvGrayMat){
    cv::Mat tmpCvMat;
    std::vector<cv::Rect> faces;
    
    NSBundle *bundle = [NSBundle bundleWithPath:[[NSBundle mainBundle] pathForResource:@"source" ofType:@"bundle"]];
    NSString *fPath = [bundle pathForResource:@"haarcascade_frontalface_alt2" ofType:@"xml"];
    if(fPath == NULL){ printf("--(!)Cant find haar model\n"); return faces; }
    
    string face_cascade_name = [fPath UTF8String];
    cv::CascadeClassifier face_cascade;
    if( !face_cascade.load(face_cascade_name) ){ printf("--(!)Error haar model loading\n"); return faces; };
    cv::equalizeHist(cvGrayMat, tmpCvMat);
    int h = cvGrayMat.rows;
    int w = cvGrayMat.cols;
    int shorter = min(w,h);
    face_cascade.detectMultiScale(tmpCvMat, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(shorter/10, shorter/10));
    
    return faces;
}


/**
 kk_grounding
 通过原图，计算并创建背色块图。
 
 
 @param originMat 原图像的mat。
 @return resualt 背景色块图的mat
 */
static cv::Mat kk_grounding(cv::Mat originMat){
    //创建背景色彩
    cv::Mat tmpCvMat;
    cv::GaussianBlur(originMat, tmpCvMat, cv::Size(7,7), 1.2,1.2);
    cv::stylization(tmpCvMat, tmpCvMat, 60, 0.99);
    return tmpCvMat;
}


/**
 kk_sketch
 通过原图，计算并创建草稿图，
 主要是通过不同强度高斯的差取高频点，然后sobel强化边界。
 
 @param originMat 原图像的mat。
 @param grayMat 原图像的灰度图mat。
 @return resualt 草图的mat
 */
static cv::Mat kk_sketch(cv::Mat originMat, cv::Mat grayMat){
    cv::Mat fan,guass,edges;
    cv::Mat resualt;
    
    resualt = Mat::zeros(originMat.size(),originMat.type());
    cv::bitwise_not(grayMat,fan);
    
    cv::GaussianBlur(fan,guass,cv::Size(11,11),0,0);
    
    cv::Mat gradX,gradY;
    cv::Sobel(grayMat, gradX, CV_16S, 0, 1);
    cv::Sobel(grayMat, gradY, CV_16S, 1, 0);
    cv::convertScaleAbs(gradX, gradX);
    cv::convertScaleAbs(gradY, gradY);
    cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, edges);
    
    for(int y=0;y<guass.rows;++y){
        for(int x=0;x<guass.cols;++x){
            double b =double(guass.at<uchar>(y,x));
            double a =double(grayMat.at<uchar>(y,x));
            int tmp = int(a+a*b/(256-b));
            tmp = tmp<255?tmp:255;
            tmp -= edges.at<char>(y,x);//边界加强
            resualt.at<cv::Vec3b>(y,x)=cv::Vec3b(tmp,tmp,tmp);
        }
    }
    
    

    return resualt;
}


/**
 kk_originProcessByCvMat
 将4通道原图转为3通道
 
 @param cvMat 原图像的mat。
 @return cvTmpMat 处理后的mat
 */
static cv::Mat kk_originProcessByCvMat(cv::Mat cvMat) {
    cv::Mat cvTmpMat;
    cv::cvtColor(cvMat, cvTmpMat, CV_RGBA2RGB);
    return cvTmpMat;
}


/**
 kk_baseProcessByCvMat
 取得灰度图
 
 @param cvMat 原图像的mat。
 @return cvTmpMat 处理后的mat
 */
static cv::Mat kk_maskProcessByCvMat(cv::Mat cvMat) {
    cv::Mat cvTmpMat;
    cv::cvtColor(cvMat, cvTmpMat, CV_RGBA2GRAY);
    return cvTmpMat;
}


/**
 kk_baseProcess
 ios原图对象转为mat，并且做基本处理。
 
 @param image 原图像的IOS对象。
 @return cvTmpMat 处理后的mat
  */
+ (cv::Mat)kk_baseProcess:(UIImage*)image{
    cv::Mat cvTmpMat;
    UIImageToMat(image, cvTmpMat);
    cv::cvtColor(cvTmpMat, cvTmpMat, CV_RGB2GRAY);
    cv::GaussianBlur(cvTmpMat, cvTmpMat, cv::Size(7,7), 1.2,1.2);
    return cvTmpMat;
}


/**
 kk_baseProcessByCvMat
 对原图做基本处理
 
 @param cvMat 原图像的mat。
 @return cvTmpMat 处理后的mat
 */
static cv::Mat kk_baseProcessByCvMat(cv::Mat cvMat) {
    cv::Mat cvTmpMat;
    cv::cvtColor(cvMat, cvTmpMat, CV_RGB2GRAY);
    cv::GaussianBlur(cvTmpMat, cvTmpMat, cv::Size(7,7), 1.2,1.2);
    return cvTmpMat;
}


/**
 kk_getOriginMat
 UIImage 转化为 opencv的矩阵 cv::Mat
 
 @param image 原图像的IOS对象。
 @return cvTmpMat 原图对应的mat
 */
+ (cv::Mat)kk_getOriginMat:(UIImage*)image {
    cv::Mat cvTmpMat;
    UIImageToMat(image, cvTmpMat);
    return cvTmpMat;
}


/**
 kk_getMinRectangleRect
 获取一个区域的最小外接矩形,以左上点和右下点的形式(x1,y1,x2,y2)
 
 @param area 一个区域的坐标。。
 @return
 */
static cv::Vec4i kk_getMinRectangleRect(std::vector<cv::Point> area ){
    
    int maxX=0,minX=INT_MAX,maxY=0,minY=INT_MAX;
    for (cv::Point p :area) {
        if (p.x>maxX)maxX = p.x;
        if (p.x<minX)minX = p.x;
        if (p.y>maxY)maxY = p.y;
        if (p.y<minY)minY = p.y;
    }
    cv::Vec4i rectPoint(minX,minY,maxX,maxY);
    return rectPoint;
}


/**
 kk_tracerPoint
 以联通方式跟踪点，取出关健线，用于形成绘画笔触。
 通过参数ceng控制一条线的长度（相当于递归深度）。
 为了提高效率，线图并未以形参形式传递，
 所以对于原线图lineMat，过程中会被修改。被取出的点将被设置为0。
 在递归跟踪线条的过程中，遇到分支线条时，将会以相对偏左向的线条为优先，取到底或达到长度限制，完成后再回头取分支
 
 @param lineMat 图片严格边界图。
 @param edgeRect 当前处理区域范围。
 @param startX 本次起点x坐标。
 @param startY 本次起点y坐标。
 @param resVec 用于存储提取结果对象。
 @param ceng 递归层数控制变量，用于避免一条线过长。
 @return
 */
static void kk_tracerPoint(cv::Mat *lineMat,cv::Vec4i *edgeRect,int startX,int startY,std::vector<cv::Point> *resVec,int ceng){
    cv::Mat &mat = *lineMat;
    cv::Vec4i &rect = *edgeRect;
    std::vector<cv::Point> &res = *resVec;
    int x;
    int y;
//    printf("到达%d",ceng);
    if (ceng > 380) return;//防止爆栈，IOS在子线程中，本函数大约430层后会爆栈
    
    //////// 左下
    x = startX-1;
    y = startY+1;
    if (x<rect[0]) return;
    if (y>rect[3]) return;

    if(mat.at<uchar>(y,x)==255){
        res.push_back(cv::Point(x,y));
        mat.at<uchar>(y,x) = 0;
//        printf("get左下:%d,%d \n",x,y);
        kk_tracerPoint(lineMat, edgeRect, x, y, resVec,ceng+1);
    }
    //////// 下
    x = startX;
    y = startY+1;
    if (y>rect[3]) return;
    if(mat.at<uchar>(y,x)==255){
        res.push_back(cv::Point(x,y));
        mat.at<uchar>(y,x) = 0;
//        printf("get:下%d,%d \n",x,y);
        kk_tracerPoint(lineMat, edgeRect, x, y, resVec,ceng+1);
    }
    //////// 右
    x = startX+1;
    y = startY;
    if (x>rect[2]) return;
    if(mat.at<uchar>(y,x)==255){
        res.push_back(cv::Point(x,y));
        mat.at<uchar>(y,x) = 0;
//        printf("get:右%d,%d \n",x,y);
        kk_tracerPoint(lineMat, edgeRect, x, y, resVec,ceng+1);
    }
    //////// 右下
    x = startX+1;
    y = startY+1;
    if (x>rect[2]) return;
    if (y>rect[3]) return;
    if(mat.at<uchar>(y,x)==255){
        res.push_back(cv::Point(x,y));
        mat.at<uchar>(y,x) = 0;
//        printf("get:右下%d,%d \n",x,y);
        kk_tracerPoint(lineMat, edgeRect, x, y, resVec,ceng+1);
    }
}


/**
 kk_pureDrawingPointAndOrder
 获得每个区块内的线条绘制点（含绘图顺序），每组线与传入的区块index对应。lineMat必须为二值线图，例如canny。
 尽量控制在800个区、30万个点内，否则会发生效率问题。
 
 @param lineMat 图片严格边界图。
 @param areas 区块序列。
 @param imageWidth 原图宽度。
 @param imageHeight 原图高度。
 @return drawingPointsVec 每个区块对应的线条绘制点顺序
 */
static std::vector<std::vector<cv::Point>> kk_pureDrawingPointAndOrder(cv::Mat lineMat, std::vector<std::vector<cv::Point>> areas, int imageWidth, int imageHeight){

    Mat dst;
    std::vector<std::vector<cv::Point>> tmpVecsInVec;//提高效率，避免重复创建
    tmpVecsInVec.push_back(areas[0]);
    cv::Mat roi = Mat::zeros(lineMat.size(),CV_8U);


    std::vector<std::vector<cv::Point>> drawingPointsVec;
    for(std::vector<cv::Point> area:areas){
        tmpVecsInVec[0]=area;
        cv::drawContours(roi,tmpVecsInVec,0,Scalar::all(255),-1);
        lineMat.copyTo(dst,roi);
        //因为roi是共用的，所以要还原回去，否则下一轮如果rect和本轮有交叉的话也许会有问题，虽然不会出现什么大错
        cv::drawContours(roi,tmpVecsInVec,0,Scalar::all(0),-1);
        std::vector<cv::Point> drawingPoints;

        //计算区域最小范围，缩小搜寻范围
        cv::Vec4i rectPoint = kk_getMinRectangleRect(area);
        //NSLog(@"rect:x：%d,y：%d，x右下：%d,y右下：%d \n",rectPoint[0],rectPoint[1],rectPoint[2],rectPoint[3]);
        int width = rectPoint[2]-rectPoint[0];
        int height = rectPoint[3]-rectPoint[1];
        if(width < imageWidth/3){
            //当区块长度不足整体图像宽度xxx时，从上到下扫描
            for (int y=rectPoint[1]; y<=rectPoint[3]; ++y)
                for (int x=rectPoint[0]; x<=rectPoint[2]; ++x){
                    if(dst.at<uchar>(y,x)==255){
                        drawingPoints.push_back(cv::Point(x,y));
                        dst.at<uchar>(y,x) = 0;//printf("start:%d,%d \n",x,y);
                        kk_tracerPoint(&dst,&rectPoint,x,y,&drawingPoints,0);//printf("\n\n\n\n\n");
                    }
                }
        }else{
            //当区块长度不足整体图像宽度xxx时，先左边再右边
            //当区块高度不足整体图像高xxx时，从上到下，超过时从中上扫到底，再补上部分
            if (height < imageHeight/2) {
                //左边
                for (int y=rectPoint[1]; y<=rectPoint[3]; ++y)
                    for (int x=rectPoint[0]; x<=rectPoint[2]*0.6; ++x){
                        if(dst.at<uchar>(y,x)==255){
                            drawingPoints.push_back(cv::Point(x,y));
                            dst.at<uchar>(y,x) = 0;//printf("start:%d,%d \n",x,y);
                            kk_tracerPoint(&dst,&rectPoint,x,y,&drawingPoints,0);//printf("\n\n\n\n\n");
                        }
                    }
                //you边
                for (int y=rectPoint[1]; y<=rectPoint[3]; ++y)
                    for (int x=rectPoint[2]*0.6; x<=rectPoint[2]; ++x){
                        if(dst.at<uchar>(y,x)==255){
                            drawingPoints.push_back(cv::Point(x,y));
                            dst.at<uchar>(y,x) = 0;//printf("start:%d,%d \n",x,y);
                            kk_tracerPoint(&dst,&rectPoint,x,y,&drawingPoints,0);//printf("\n\n\n\n\n");
                        }
                    }
            }else{
                //当区块高度不足整体图像高xxx时，从上到下，超过时从中上扫到底，再补上部分
                //左下边
                for (int y=rectPoint[1]*1.3; y<=rectPoint[3]; ++y)
                    for (int x=rectPoint[0]; x<=rectPoint[2]*0.6; ++x){
                        if(dst.at<uchar>(y,x)==255){
                            drawingPoints.push_back(cv::Point(x,y));
                            dst.at<uchar>(y,x) = 0;//printf("start:%d,%d \n",x,y);
                            kk_tracerPoint(&dst,&rectPoint,x,y,&drawingPoints,0);//printf("\n\n\n\n\n");
                        }
                    }
                //右下边
                for (int y=rectPoint[1]*1.3; y<=rectPoint[3]; ++y)
                    for (int x=rectPoint[2]*0.6; x<=rectPoint[2]; ++x){
                        if(dst.at<uchar>(y,x)==255){
                            drawingPoints.push_back(cv::Point(x,y));
                            dst.at<uchar>(y,x) = 0;//printf("start:%d,%d \n",x,y);
                            kk_tracerPoint(&dst,&rectPoint,x,y,&drawingPoints,0);//printf("\n\n\n\n\n");
                        }
                    }
                //左上边
                for (int y=rectPoint[1]; y<=rectPoint[3]*0.3; ++y)
                    for (int x=rectPoint[0]; x<=rectPoint[2]*0.6; ++x){
                        if(dst.at<uchar>(y,x)==255){
                            drawingPoints.push_back(cv::Point(x,y));
                            dst.at<uchar>(y,x) = 0;//printf("start:%d,%d \n",x,y);
                            kk_tracerPoint(&dst,&rectPoint,x,y,&drawingPoints,0);//printf("\n\n\n\n\n");
                        }
                    }
                //右上边
                for (int y=rectPoint[1]; y<=rectPoint[3]*0.3; ++y)
                    for (int x=rectPoint[2]*0.6; x<=rectPoint[2]; ++x){
                        if(dst.at<uchar>(y,x)==255){
                            drawingPoints.push_back(cv::Point(x,y));
                            dst.at<uchar>(y,x) = 0;//printf("start:%d,%d \n",x,y);
                            kk_tracerPoint(&dst,&rectPoint,x,y,&drawingPoints,0);//printf("\n\n\n\n\n");
                        }
                    }
            }
            
        }
        
        drawingPointsVec.push_back(drawingPoints);
    }
    
    return drawingPointsVec;
}


/**
 kk_drawLineSubProcess
 绘制草图的处理过程。
 包括绘制至屏幕。
 通过笔触点，更新全局绘图上下文空间。
 通过frameCount，控制每次图像渲染以及更新至屏幕的间隔。
 
 @param pointsVec 草图绘制笔触顺序。
 @param sketchMat 草图。
 @param frameCount 当前绘制过程的总帧数。
 @return
 */
-(void)kk_drawLineSubProcess:(std::vector<cv::Point>*)pointsVec sketchMat:(cv::Mat*) sketchMat frameCount:(int)frameCount{
    //绘制线条。
    std::vector<cv::Point> &points = *pointsVec;
    int pointCount = int(points.size());
    
    int pointsCountInFrame = pointCount / frameCount;
    printf("pointCount:%d,%d,%d,\n",pointCount,frameCount,pointsCountInFrame);
    
    
    std::vector<cv::Point> draw(pointsCountInFrame);
    for (int i=0; i<frameCount+1; ++i) {
        if(!self.bDrawing)return ;
        if (i==frameCount) {//收尾，最后一部分是小于一帧所需点数量的
            std::vector<cv::Point> theTailPoints;
            for(int tmp=pointsCountInFrame*i; tmp<pointCount; ++tmp){
                theTailPoints.push_back(points[tmp]);
            }
            if(!self.bDrawing)return ;
            [self kk_updateImageBuffByPixelsVec:&theTailPoints sketchDraft:sketchMat];
            [self updateImage];
            //                    printf("draw[%d]:%lu\n",i,theTailPoints.size());
            break;
        }
        std::copy(points.begin() + pointsCountInFrame*i,
                  points.begin() + pointsCountInFrame*(i+1),
                  draw.begin());
        if(!self.bDrawing)return ;
        [self kk_updateImageBuffByPixelsVec:&draw sketchDraft:sketchMat];
        if(!self.bDrawing)return ;
        [self updateImage];
        //printf("draw[%d]:%lu,%d\n",i,draw.size(),pointsCountInFrame*(i+1));
        if(!self.bDrawing)return;
        [NSThread sleepForTimeInterval:1.f/self.fps];
    }
}


/**
 pureDaubingGroundingPoints
 按照区块顺序，提取背景上色的画刷笔触点以及顺序。
 
 @param areasMat 区块序列。
 @param imageWidth 原图宽度。
 @param imageheight 原图高度。
 @return 区块对应的上色笔触顺序。
 */
static std::vector<std::vector<cv::Point>> pureDaubingGroundingPoints(std::vector<std::vector<cv::Point>> areasMat, int imageWidth, int imageheight){
    std::vector<std::vector<cv::Point>> areas = areasMat;
    std::vector< std::vector<cv::Point>> daubingPointsVec;
    
    for(std::vector<cv::Point> area:areas){
        //计算区域最小范围，缩小搜寻范围
        cv::Vec4i rectPoint = kk_getMinRectangleRect(area);
//        NSLog(@"rect:x：%d,y：%d，x右下：%d,y右下：%d \n",rectPoint[0],rectPoint[1],rectPoint[2],rectPoint[3]);
        
        //由于区块计算会全图的两边少一些像素，所以补上
        if(rectPoint[0]<2){
            rectPoint[0]-=1;
        }
        if (rectPoint[2]<imageWidth-1 && rectPoint[2]>imageWidth-3) {
            rectPoint[2]+=2;
        }
        
        int width = rectPoint[2]-rectPoint[0];
        int height = rectPoint[3]-rectPoint[1];
        int spanWidth = kk_calculateBrushWidth(area);
            width+=spanWidth;//让宽度多走一个笔刷，否则区域右边会出现锯齿。不过上色时需要防止画出界
        
        //斜线扫描区域
        std::vector<cv::Point> daubingPointsInArea;
        int scanLinesCount =width+height-1;
        int min_dim=min(height, width);
        for (int scanLineIndex=0; scanLineIndex<scanLinesCount;) {
            int scanLineLen = min(min(scanLineIndex+1, min_dim), scanLinesCount-scanLineIndex);
            int startRow = min(scanLineIndex, width-1);
            int startCol = max(0, scanLineIndex-width+1);
            for (int pointInLineIndex=0; pointInLineIndex<scanLineLen; ++pointInLineIndex){
                cv::Point2i p;
                p = cv::Point2i(rectPoint[0]+startRow-pointInLineIndex,rectPoint[1]+startCol+pointInLineIndex);
                if (scanLineIndex%2 == 0) {//正向与反向绘制
                    daubingPointsInArea.push_back(p);
                }else{
                    if (pointInLineIndex==0) {
                        daubingPointsInArea.push_back(p);
                    }else{
                        daubingPointsInArea.insert(daubingPointsInArea.end()-pointInLineIndex,p);
                    }
                    
                }
            }
            scanLineIndex+=spanWidth;
        }
        daubingPointsVec.push_back(daubingPointsInArea);

    }
    return daubingPointsVec;
}

/**
 kk_calculateBrushWidth
 根据绘画区域，计算笔刷宽度。
 默认笔刷宽度为1像素。
 通过宏定义LIMIT_AREASIZE_BRUSHWIDTH判断是否需要进行笔刷加宽
 如果当前区域宽度大于阈值时，才更新笔刷宽度。
 笔刷宽度为区域宽度的百分之比，通过宏定义RATE_BRUSHWIDTH_GROUNDING控制
 
 @param area 区块。
 @return brushWidth 笔刷宽度，为奇数。
 */
static int kk_calculateBrushWidth(std::vector<cv::Point> area){
    cv::Vec4i rectPoint = kk_getMinRectangleRect(area);
    //NSLog(@"rect:x：%d,y：%d，x右下：%d,y右下：%d \n",rectPoint[0],rectPoint[1],rectPoint[2],rectPoint[3]);
    int width = rectPoint[2]-rectPoint[0];
    int brushWidth = 1;//跳过部分，意味着绘制时笔刷的是粗细程度
    if (width > LIMIT_AREASIZE_BRUSHWIDTH){
        //假设笔刷为宽度的10%，也就是意味着10笔涂完一区域
        //spanWidth必须为单数
        brushWidth = int(width*RATE_BRUSHWIDTH_GROUNDING)%2==0?(int(width*RATE_BRUSHWIDTH_GROUNDING)+1):int(width*RATE_BRUSHWIDTH_GROUNDING);
    }
    return brushWidth;
}

/**
 comp
 区域排序的算子
 @return
 */
bool comp(cv::Vec2d &a, cv::Vec2d &b){
    return a[1] > b[1];
}

/**
 daubing
 背景色的具体上色过程
 先按照区块顺序，上每个区块的平均色作为底色。
 然后以两次不同权重依次上色，来实现水彩或油画上色效果。
 
 @param areasVec 区块序列。
 @param orderlyPointsVec 区块序列对应的笔触集合。以序列中的点控制笔刷顺序。
 @param groundingMat 原图对应的背景色，背景色可分为水彩或油画或其他效果，取决于背景处理方法。
 @param brushmaskMats 笔刷样式掩板。
 @param frameCount 当前步骤需要绘制多少帧。
 @param totalAreaSize 当前绘制步骤总共涉及的绘制面积。
 @param bSort 按默认区块顺序绘制，还是由大到小排序后再绘制。
 @param step 当前绘制的内容处于哪个步骤，具体步骤参见BUSHINGTYPY，
              主要影响笔刷上色权重。当步骤为AVG时，会对区块取均值，使用均值进行上色。
 @return
 */
-(void)daubing:(std::vector<std::vector<cv::Point>>*)areasVec orderlyPointsVec:(std::vector<std::vector<cv::Point>> *)orderlyPointsVec groundingMat:(cv::Mat*) groundingMat brushmaskMat:(cv::Mat*)brushmaskMat frameCount:(int)frameCount totalAreaSize:(double)totalAreaSize bSort:(bool)bSort step:(BUSHINGTYPY)step{
    //具体的上色！
    std::vector<std::vector<cv::Point>> &areas = *areasVec;
    std::vector<std::vector<cv::Point>> &points = *orderlyPointsVec;
    cv::Mat &grounding = *groundingMat;
    cv::Mat &brushmask = *brushmaskMat;
    
    
    std::vector<std::vector<cv::Point>> tmpVecsInVec;//提高效率，避免重复创建
    tmpVecsInVec.push_back(areas[0]);
    cv::Mat roi = Mat::zeros(grounding.size(),CV_8U);//提高效率，避免重复创建
    cv::Mat inverseROI = cv::Mat(grounding.size(),CV_8U,cv::Scalar::all(255));//提高效率，避免重复创建
    cv::Mat backgroundROI = cv::Mat(grounding.size(),CV_8UC3,cv::Scalar(255,0,255));//提高效率，避免重复创建
    

    std::vector<cv::Vec2d> sizeIndexes;
    for(int areaIndex=0; areaIndex<areas.size(); ++areaIndex){
        sizeIndexes.push_back(cv::Vec2d(areaIndex,cv::contourArea(areas[areaIndex])));
    }
    BUSHINGTYPY brushType = step;

    
    if(bSort){
        std::sort(sizeIndexes.begin(), sizeIndexes.end(), comp);
    }
    
    
    int orderIndex = -1;
    int accumulative = 1;
    for(cv::Vec2d indexVec:sizeIndexes){
        int areaIndex =indexVec[0];
        double areaSize = indexVec[1];
        
        orderIndex++;
        
        if(areaSize==0){
            continue;
        }
        
        
        //控制绘制速度
        std::vector<cv::Point> pointsInArea = points[areaIndex];
        //本次绘制区域中的引导点数量，并不是真正的绘制点，因为笔刷效果一次画的数量为笔刷宽度的2倍。
        int pointsCountInArea = int(pointsInArea.size());
        //通过本区域的面积占总体面积的比例，计算本区域应该承担整个上色动画的多少帧画面
        double areaRate = areaSize/totalAreaSize;
        double framesInThisArea = frameCount * areaRate;
        framesInThisArea = framesInThisArea<1?1:framesInThisArea;
        int pointsInFrame = int(pointsCountInArea/framesInThisArea);//一帧内应该包含多少绘图引导点。既笔刷的中心点。
        
//        printf("[%d]:%d, %.1f, %f\n",orderIndex,areaIndex,areaSize,areaRate);
        if(orderIndex>800 && areaSize<150 && areaRate<.0005){
            printf("大量小块不再处理直接跳过");
            break;//大量小块不再处理直接跳过。
        }
        
        Mat dst;//真正的绘制目标区域；
        
        //提取绘制区域内容到dst
        tmpVecsInVec[0] = areas[areaIndex];
        
        //设置拷贝上色区
        cv::drawContours(roi,tmpVecsInVec,0,Scalar::all(255),-1);
        grounding.copyTo(dst,roi);
        cv::Scalar avgChannels;
        if (step == AVG)
            avgChannels =cv::mean(dst,roi);
        
        cv::drawContours(roi,tmpVecsInVec,0,Scalar::all(0),-1);
        
        //设置背景色
        cv::drawContours(inverseROI,tmpVecsInVec,0,Scalar::all(0),-1);
        backgroundROI.copyTo(dst,inverseROI);
        cv::drawContours(inverseROI,tmpVecsInVec,0,Scalar::all(255),-1);

        
        int brushWidth = kk_calculateBrushWidth(tmpVecsInVec[0]);
        cv::Vec4i drawRect = kk_getMinRectangleRect(tmpVecsInVec[0]);
        
        std::vector<cv::Point> truelyBrushCompensatePoints;
        std::vector<cv::Vec3b> truelyBrushCompensateColors;

        
        
        for (int i=0; i<pointsInArea.size(); ++i) {
            cv::Point startP = pointsInArea[i];
            for (int j=0; j<brushWidth; j++) {//如果笔刷大于1，将横向读取产生笔刷效果
                //从引导点往左一个笔刷宽度开始，画2倍笔刷。多出的部分用边界控制，实际上并未产生多少浪费。如果1倍笔刷，在边界上会存在遗漏，形成锯齿。
                cv::Point p = cv::Point(startP.x-j,startP.y);
                if (p.x > drawRect[2]+1)continue;//控制越界，减少性能浪费
                if (p.x < drawRect[0]-1||p.x<0)continue;//控制越界，减少性能浪费
                Vec3b pixelVec3b = dst.at<Vec3b>(p);
                //赤红背景不进行绘制，因为区域实际上是多边形，但是绘制为了方便和效率，实际上取的是矩形，多余部分不取
                if(pixelVec3b[0]!=255&&pixelVec3b[1]!=0&&pixelVec3b[2]!=255){
                    Vec3b pixel;
                    if (areaRate>LIMIT_TRUE_BRUSH_AREARATE) {//触发真实笔刷
                        uchar mask = brushmask.at<uchar>(p);
                        if (mask > 5) {
                            //掩版高亮部分绘制
                            if(step == AVG)
                                pixel = Vec3b(char(int(avgChannels.val[0])),char(int(avgChannels.val[1])),char(int(avgChannels.val[2])));
                            else
                                pixel = Vec3b(pixelVec3b[0],pixelVec3b[1],pixelVec3b[2]);
                            kk_updateImageBuffByPixelVec3b(&p, &pixel, _pRgbImageBuf, _imageWidth, brushType);
                        }else{
                            //掩版俺部绘制存储，后补绘制。
                            if(step == AVG)
                                pixel = Vec3b(char(int(avgChannels.val[0])),char(int(avgChannels.val[1])),char(int(avgChannels.val[2])));
                            else
                                pixel = Vec3b(pixelVec3b[0],pixelVec3b[1],pixelVec3b[2]);
                            truelyBrushCompensatePoints.push_back(p);
                            truelyBrushCompensateColors.push_back(pixel);
                        }
                    }else{
                        //为了节省效率，避免mask在不需要的时候被读取，重复一遍以下逻辑
                        if(step == AVG)
                            pixel = Vec3b(char(int(avgChannels.val[0])),char(int(avgChannels.val[1])),char(int(avgChannels.val[2])));
                        else
                            pixel = Vec3b(pixelVec3b[0],pixelVec3b[1],pixelVec3b[2]);
                        kk_updateImageBuffByPixelVec3b(&p, &pixel, _pRgbImageBuf, _imageWidth, brushType);
                    }
                }
                accumulative++;
            }

            //////
            if(i!=0 && (i % pointsInFrame) == 0){
                [self updateImage];
                [NSThread sleepForTimeInterval:1.f/self.fps];
            }else if (i==pointsInArea.size()-1){
                if (areas.size()>600 && areaRate<0.001) {//控制背景过于复查的情况，小细节就不一直占用帧数了。
                    if ((accumulative % pointsInFrame/2) == 0) {
                        accumulative = 1;
                        [self updateImage];
                        //[NSThread sleepForTimeInterval:(1.f/self.fps)*(areaRate+0.2)];//防止最后一些非常小的区域占用太多帧数
                    }else{
                        accumulative++;
                    }
                }else{
                    [self updateImage];
                    [NSThread sleepForTimeInterval:(1.f/self.fps)*(areaRate+0.2)];//防止最后一些非常小的区域占用太多帧数
                }
            }
        }
        //真实笔刷会有遗漏点，当绘图过半时进行补偿
        if (areaRate>0.3 && truelyBrushCompensatePoints.size()>0) {
            for (int comIdx=0; comIdx<truelyBrushCompensatePoints.size(); ++comIdx) {
                cv::Point comP = truelyBrushCompensatePoints[comIdx];
                Vec3b comPixel = truelyBrushCompensateColors[comIdx];
                kk_updateImageBuffByPixelVec3b(&comP, &comPixel, _pRgbImageBuf, _imageWidth, brushType);
                if( (comIdx!=0 && (comIdx % (pointsInFrame*brushWidth))==0) || comIdx==pointsInArea.size()-1){
                    [self updateImage];
                    [NSThread sleepForTimeInterval:1.f/self.fps];
                }
            }
        }
        //真实笔刷end
    }
    [self updateImage];//函数结束前再更新一下避免遗漏。
}


/**
 daubGroundingSubProcess
 初次上色，具体分为3个步骤，
 先按照区块顺序，上每个区块的平均色作为底色。
 然后以两次不同权重依次上色，来实现水彩或油画上色效果。
 
 并且通过区域计算以及宏定义预设绘图步骤时间，控制每次图像渲染以及更新至屏幕的间隔。

 
 @param areasVec 区块序列
 @param orderlyPointsVec 区块序列对应的笔触集合。以序列中的点控制笔刷顺序
 @param groundingMat 原图对应的背景色，背景色可分为水彩或油画或其他效果，取决于背景处理方法。
 @param brushmaskMats 笔刷样式蒙版
 @return
 */
-(void)daubGroundingSubProcess:(std::vector<std::vector<cv::Point>> *)areasVec orderlyPointsVec:(std::vector<std::vector<cv::Point>> *)orderlyPointsVec groundingMat:(cv::Mat*) groundingMat brushmaskMats:(cv::Mat*)brushmaskMats{
    //涂抹草图底色
    std::vector<std::vector<cv::Point>> &areas = *areasVec;
    cv::Mat *brushes = brushmaskMats;
    //计算绘图速度控制变量
    int frameCount_avg = DURATION_AVG_GROUNDING * self.groundingDurationMultiplying * self.fps;
    double totalAreaSize=0.f;
    for(int areaIndex=0; areaIndex<areas.size(); ++areaIndex){
        totalAreaSize += cv::contourArea(areas[areaIndex]);
    }
    
    //绘制大片背景底色
    [self daubing:areasVec orderlyPointsVec:orderlyPointsVec groundingMat:groundingMat brushmaskMat:&(brushes[arc4random_uniform(BRUSH_COUNTS)]) frameCount:frameCount_avg totalAreaSize:totalAreaSize bSort:true step:AVG];
    //绘制背景底色
    int frameCount_grounding = DURATION_GROUNDING * self.groundingDurationMultiplying * self.fps;
    [self daubing:areasVec orderlyPointsVec:orderlyPointsVec groundingMat:groundingMat brushmaskMat:&(brushes[arc4random_uniform(BRUSH_COUNTS)]) frameCount:frameCount_grounding totalAreaSize:totalAreaSize bSort:false step:GROUNDING1];
    [self daubing:areasVec orderlyPointsVec:orderlyPointsVec groundingMat:groundingMat brushmaskMat:&(brushes[arc4random_uniform(BRUSH_COUNTS)]) frameCount:frameCount_grounding totalAreaSize:totalAreaSize bSort:true step:GROUNDING2];
}

/**
 daubOriginSubProcess
 背景上色完成后的步骤，把原图按照不同权重，覆盖草图。
 @param areasVec 区块序列
 @param orderlyPointsVec 区块序列对应的笔触集合。以序列中的点控制笔刷顺序
 @param originMat 原始图像
 @param brushmaskMats 笔刷样式蒙版
 @param bIsEnding 原图绘制分为两步，当ending为YES时，以高权重覆盖草图，为NO时，地权重覆盖草图
 @return
 */
-(void)daubOriginSubProcess:(std::vector<std::vector<cv::Point>> *)areasVec orderlyPointsVec:(std::vector<std::vector<cv::Point>> *)orderlyPointsVec originMat:(cv::Mat*) originMat brushmaskMats:(cv::Mat*)brushmaskMats bIsEnding:(bool)bIsEnding{
    //涂抹原图
    std::vector<std::vector<cv::Point>> &areas = *areasVec;
    cv::Mat *brushes = brushmaskMats;
    
    //计算绘图速度控制变量
    int frameCount = DURATION_ORIGIN * self.originDurationMultiplying * self.fps;
    double totalAreaSize=0.f;
    for(int areaIndex=0; areaIndex<areas.size(); ++areaIndex){
        totalAreaSize += cv::contourArea(areas[areaIndex]);
    }
    
    if (!bIsEnding) {
        //原图前的图层
        [self daubing:areasVec orderlyPointsVec:orderlyPointsVec groundingMat:originMat brushmaskMat:&(brushes[arc4random_uniform(BRUSH_COUNTS)]) frameCount:frameCount totalAreaSize:totalAreaSize bSort:false step:ORIGINIMAGE1];
    }else{
            //绘制原图
        [self daubing:areasVec orderlyPointsVec:orderlyPointsVec groundingMat:originMat brushmaskMat:&(brushes[arc4random_uniform(BRUSH_COUNTS)]) frameCount:frameCount totalAreaSize:totalAreaSize bSort:true step:ORIGINIMAGE2];
    }
}


/**
 finalDealSubProcess
 绘图的最终步骤，原图覆盖
 @param brushmaskMats 笔刷样式的蒙版
 @return
 */

-(void)finalDealSubProcess:(cv::Mat*)originMat brushmaskMats:(cv::Mat*)brushmaskMats {
    //涂抹原图
    cv::Mat &origin = *originMat;
    cv::Mat *brushes = brushmaskMats;
    
    //计算绘图速度控制变量
    int frameCount = DURATION_FINAL * self.fps;
    std::vector<std::vector<cv::Point>> areas;
    std::vector<cv::Point>area;
    //由于区域计算不含边，涂抹控制时把多边形的左右两边包含上了。
    //所以最后一步用全图覆盖时，不能以0点开始，不能以完整宽度结束，因为图像最左边和最右边一列正常情况在边界上。
    area.push_back(cv::Point(0,0));
    area.push_back(cv::Point(origin.cols-1,0));
    area.push_back(cv::Point(origin.cols-1,origin.rows-1));
    area.push_back(cv::Point(0,origin.rows-1));
    areas.push_back(area);
    std::vector<std::vector<cv::Point>> drawPoints = pureDaubingGroundingPoints(areas,origin.cols,origin.rows);
    
    [self daubing:&areas orderlyPointsVec:&drawPoints groundingMat:originMat brushmaskMat:&(brushes[arc4random_uniform(BRUSH_COUNTS)]) frameCount:frameCount totalAreaSize:origin.cols*origin.rows bSort:false step:FINAL];
}


/**
 drawingProcess
 完整的绘图过程。
 
 @return
 */
-(void)drawingProcess{
    self.bDrawing = YES;
    cv::Mat originImgage = [UIPaintingImageView kk_getOriginMat:self.image];
    srand(time_t(0));//for随即笔刷的种子
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT,0), ^{
        while (1) {
            cv::Mat originMat = kk_originProcessByCvMat(originImgage);
            cv::Mat baseProcessedMat = kk_baseProcessByCvMat(originMat);
            std::vector<cv::Rect> faces = kk_detectFaceByhaar(baseProcessedMat);
            cv::Mat edgesMat = kk_cannyKenel(baseProcessedMat);
            cv::Mat sketchDraft = kk_sketch(originMat,baseProcessedMat);
            std::vector<std::vector<cv::Point>> areas = kk_cutArea(baseProcessedMat, originMat, faces, self->_imageWidth,self->_imageHeight);
            std::vector<std::vector<cv::Point>> drawingPointsVec = kk_pureDrawingPointAndOrder(edgesMat,areas,self->_imageWidth,self->_imageHeight);
            
            //加载笔刷
            cv::Mat brushmaskMats[BRUSH_COUNTS];
            for (int i=0; i<BRUSH_COUNTS; ++i) {
                brushmaskMats[i] = kk_maskProcessByCvMat([UIPaintingImageView kk_getOriginMat:self.brushMaskImageArray[i]]);
                cv::resize(brushmaskMats[i], brushmaskMats[i], originMat.size(),0,0,INTER_LINEAR);
            }
 
            
            std::vector<cv::Point> drawPixelsVec;
            for(std::vector<cv::Point> pointsVec :drawingPointsVec){
                for(cv::Point p :pointsVec){
                    drawPixelsVec.push_back(p);
                }
            }
            //画线稿时开始生成背景色，这玩意非常耗时。短边800需0.8s，1800需4s 2500需8～16s
            __block cv::Mat groundingMat;
            __block std::vector<std::vector<cv::Point>> daubingPointsVec;
            if(self.groundingDurationMultiplying != 0){
                dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT,0), ^{
                    groundingMat = kk_grounding(originMat);
                    NSLog(@"groundingMat 创建完成");
                });
            }
            
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT,0), ^{
                daubingPointsVec = pureDaubingGroundingPoints(areas, self->_imageWidth, self->_imageHeight);
                NSLog(@"daubingPointsVec 创建完成");
            });
            
            if(self.sketchDurationMultiplying != 0)
                [self kk_drawLineSubProcess:&drawPixelsVec sketchMat:&sketchDraft frameCount:DURATION_SKETCH * self.sketchDurationMultiplying * self.fps];
            if(!self.bDrawing)return;
            
            if(self.groundingDurationMultiplying != 0)
                while(groundingMat.empty() || daubingPointsVec.empty()){
                    if (!self.bDrawing)break;
                    NSLog(@"wait");
                    [NSThread sleepForTimeInterval:0.35f];
                }
            
            CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
            
            if(self.groundingDurationMultiplying != 0)
                [self daubGroundingSubProcess:&areas orderlyPointsVec:&daubingPointsVec groundingMat:&groundingMat brushmaskMats:brushmaskMats];
            if(!self.bDrawing)return;
            printf("daubGroundingSubProcess end \n");
            
            if(self.originDurationMultiplying != 0)
                [self daubOriginSubProcess:&areas orderlyPointsVec:&daubingPointsVec originMat:&originMat brushmaskMats:brushmaskMats bIsEnding:false];
            if(!self.bDrawing)return;
            printf("daubOriginSubProcess end \n");
            
            if(self.lineDurationMultiplying != 0)
                [self kk_drawLineSubProcess:&drawPixelsVec sketchMat:NULL frameCount:DURATION_GROUNDING * self.lineDurationMultiplying * self.fps];
            if(!self.bDrawing)return;
            printf("kk_drawLineSubProcess end \n");
            
            if(self.originDurationMultiplying != 0)
                [self daubOriginSubProcess:&areas orderlyPointsVec:&daubingPointsVec originMat:&originMat brushmaskMats:brushmaskMats bIsEnding:true];
            printf("daubOriginSubProcess2 end \n");
            
            if(self.originDurationMultiplying != 0)
                [self finalDealSubProcess:&originMat brushmaskMats:brushmaskMats];
            printf("finalDealSubProcess end \n");
            
            CFAbsoluteTime end = CFAbsoluteTimeGetCurrent();
            NSLog(@"%f", end - start);
            
            if ([self.delegate respondsToSelector:@selector(drawEnding)]  &&  self.delegate != nil){
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self.delegate drawEnding];
                });
            }
            
            printf("end");
            break;
        }
    });
}

/**
 startDrawing  -public
 开始绘图

 @return
 */
- (void)startDrawing{
    NSAssert(self.isPrepare == YES, @"请使用prepareForPainting方法进行绘图前准备");
//    self.bDrawing = YES;
//    cv::Mat originMat = kk_originProcessByCvMat([UIPaintingImageView kk_getOriginMat:self.image]);
//    cv::Mat baseProcessedMat = kk_baseProcessByCvMat(originMat);
//    std::vector<cv::Rect> faces = kk_detectFaceByhaar(baseProcessedMat);
//    cv::Mat edgesMat = kk_cannyKenel(baseProcessedMat);
//    std::vector<std::vector<cv::Point>> areas = kk_cutArea(baseProcessedMat, originMat, faces);
//
//    std::vector<std::vector<cv::Point>> drawingPointsVec = kk_pureDrawingPointAndOrder(edgesMat,areas);
//
//    cv::Mat tmp = edgesMat;
//    cv::Mat sketchDraft = kk_sketch(originMat,baseProcessedMat);
    
//    CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
//    cv::Mat groundingMat = kk_grounding(originMat);
//    CFAbsoluteTime end = CFAbsoluteTimeGetCurrent();
//    NSLog(@"%f", end - start);
    
    [self drawingProcess];
    
//    cv::Mat Kmeans = splitByKmeans(&originMat);

//    NSData * imgData =[NSData dataWithContentsOfFile:@"/Users/kk/Desktop/mask.jpg"];
//    cv::Mat brushmaskMat = kk_originProcessByCvMat([UIPaintingImageView _getOriginMat:[UIImage imageWithData:imgData]]);
//    cv::resize(brushmaskMat, brushmaskMat, originMat.size());
//    cv::Mat show = originMat + brushmaskMat;
//
//    cv::Mat roi = originMat(faces[0]);
//    cvAvg区域平均值
//    pRgbImageBuf
//    self.image = [UIPaintingImageView imageFromCVMat:Kmeans];

//    self.image = [UIPaintingImageView imageEdgesFromImage:self.image];
    
}


/**
 kk_updateImageBuffByPixelVec3b
 根据不同步骤，选择不同阈值，最终表现为不同步骤的上色强度不同。
 向_pRgbImageBuf上绘制像素，上下文渲染h与绘制并不同步。
 
 @param point 绘制点坐标
 @param rgbVec3b 将要上色的rpg值。
 @param pRgbBuf 绘制目标控件的指针，申请绘图上下文时产生的数组空间。
 @param imageWidth 绘图对象宽度。
 @param type 对应的绘图步骤。
 @return
 */
static void kk_updateImageBuffByPixelVec3b(cv::Point* point, Vec3b* rgbVec3b, uint32_t* pRgbBuf,int imageWidth, BUSHINGTYPY type){
    uint32_t* pCurPtr = pRgbBuf;
    uint8_t* ptr = (uint8_t*)(pCurPtr+imageWidth*(*point).y+(*point).x);
    float threshold = .0f;
    switch (type) {
        case NORMAL:
            break;
        case AVG:
            threshold = .7f;
            break;
        case GROUNDING1:
            threshold = .8f;
            break;
        case GROUNDING2:
            threshold = .4f;
            break;
        case ORIGINIMAGE1:
            threshold = .8f;
            break;
        case ORIGINIMAGE2:
            threshold = .1f;
            break;
        case FINAL:
            threshold = .0f;
            break;
        default:
            assert("kk_updateImageBuffByPixelVec3b:????,type? \n");
            break;
    }
    ptr[3] = ptr[3]*threshold+(*rgbVec3b)[0]*(1-threshold);
    ptr[2] = ptr[2]*threshold+(*rgbVec3b)[1]*(1-threshold);
    ptr[1] = ptr[1]*threshold+(*rgbVec3b)[2]*(1-threshold);

}

/**
 kk_updateImageBuffByPixelsVec
 向_pRgbImageBuf上绘制像素，上下文渲染h与绘制并不同步。
 用于绘制线条。
 
 @param pointsVec 线条的绘制点序列
 @param sketchDraft 当本参数为空时，将按照固定值，直接更新1像素内容。
                    当参数不为空时，将读取Mat进行点绘制。
                    注意函数中的变量lineSize，相当于绘制宽度。
                    pointsVec作为中心点，按宽度绘制。
 @return
 */
-(void)kk_updateImageBuffByPixelsVec:(std::vector<cv::Point> *)pointsVec sketchDraft:(cv::Mat *)sketchDraft{
    
    std::vector<cv::Point> &points = *pointsVec;
    cv::Mat &sketch = *sketchDraft;
    if (sketchDraft == NULL) {
        //当未传sketchDraft时，绘图只按照坐标绘制实线
        for (cv::Point p: points) {
            uint32_t* pCurPtr = _pRgbImageBuf;
            pCurPtr = pCurPtr + _imageWidth*p.y+p.x;
            uint8_t* ptr = (uint8_t*)pCurPtr;
            ptr[3] = 60;//R
            ptr[2] = 60;//G
            ptr[1] = 60;//B //ptr[0] = 255;//A 初始化时已赋值为255
        }
    }else{//当sketchDraft不为空时，绘制草稿
        uint8_t lineSize = 7 ; //线条粗细必须大于0，且为单数，尽量不要太大，效率低。参考：ipx最好<=13
        uint8_t loopSize = uint8_t(lineSize-1)/2;
        for (cv::Point p: points) {
            for(int xi=(-loopSize);xi<=loopSize;++xi){
                for(int yi=(-loopSize);yi<=loopSize;++yi){
                    int x = p.x + xi, y = p.y + yi;
                    if (x<0||x>=_imageWidth)continue;
                    if (y<0||y>=_imageHeight)continue;
                    Vec3b ske = sketch.at<Vec3b>(y,x);
                    uint32_t* pCurPtr = _pRgbImageBuf;
                    pCurPtr = pCurPtr+_imageWidth*y+x;
                    uint8_t* ptr = (uint8_t*)pCurPtr;
                    ptr[3] = ske[0];//R
                    ptr[2] = ske[1];//G
                    ptr[1] = ske[2];//B
                }
            }
            
        }
    }

}

/**
 updateImage
 根据_pRgbImageBuf渲染图片，并加载至屏幕
 
 @param
 @return
 */
-(void)updateImage{
    dispatch_async(dispatch_get_main_queue(), ^{
        CGImageRef imageRef = CGImageCreate(self.imageWidth, self.imageHeight, 8, 32, self.bytesPerRow, self.colorSpace,
                                            kCGImageAlphaLast | kCGBitmapByteOrder32Little, self.dataProvider,
                                            NULL, true, kCGRenderingIntentDefault);
        self.image = [UIImage imageWithCGImage:imageRef];
        CGImageRelease(imageRef);
    });
}

/**
 setImage
 重载父类函数，主要作用于初始化后，准备数据时，重采样图片。
 
 @param image 需要进行painting的原图
 @return
 */
- (void)setImage:(UIImage *)image{
    if(image == nil){
        [super setImage:image];
        return;
    }
    UIImage *tmp = image;
    if(!self.bDrawing){
        short w = image.size.width;
        short h = image.size.height;
        short shorter = min(w,h);
        if(shorter > LIMIT_ORIGINSIZE_F){
            float scale = float(LIMIT_ORIGINSIZE_F)/shorter;
            UIGraphicsBeginImageContext(CGSizeMake(w * scale, h * scale));
            NSLog(@"scale:%f,%f,%f",scale,w * scale,h * scale);
            [image drawInRect:CGRectMake(0, 0, w * scale, h * scale)];
            tmp = UIGraphicsGetImageFromCurrentImageContext();
            UIGraphicsEndImageContext();
        }
        NSLog(@"shoter:%d,%f,%f",shorter,tmp.size.width,tmp.size.height);
    }
    [super setImage:tmp];
}


/**
 prepareForPainting  -public
 本对象初始化后必须执行的方法。用于设置各种参数以及初始化绘图上下文。
 
 @param img 需要进行painting的原图
 @param fps 每秒理论绘制过程每秒帧数参考值，并不严格准守。
 @param sketchTimeMulti 线稿绘制过程时长倍率参考值，并不严格准守。当设置为0时，该绘图步骤跳过。
 @param groundingTimeMulti 背景上色过程时长倍率参考值，并不严格准守。当设置为0时，该绘图步骤跳过。
 @param lineTimeMulti 补线绘制过程时长倍率参考值，并不严格准守。当设置为0时，该绘图步骤跳过。
 @param originTimeMulti 原图覆盖绘制过程时长倍率参考值，并不严格准守。当设置为0时，该绘图步骤跳过。
 @return
 */
-(void)prepareForPainting:(UIImage*)img fps:(uint)fps sketchTimeMulti:(float)stm groundingTimeMulti:(float)gtm lineTimeMulti:(float)ltm originTimeMulti:(float)otm{
    
    [self clean];
    
    self.image = img;
    
    self.sketchDurationMultiplying = stm;
    self.groundingDurationMultiplying = gtm;
    self.lineDurationMultiplying = ltm;
    self.originDurationMultiplying = otm;
    
    self.colorSpace = CGColorSpaceCreateDeviceRGB();
    CGColorSpaceRetain(self.colorSpace);
    self.imageWidth = self.image.size.width;
    self.imageHeight = self.image.size.height;
    self.bytesPerRow = _imageWidth * 4;
    self.pRgbImageBuf = (uint32_t*)malloc(_bytesPerRow * _imageHeight);
    CGContextRef context = CGBitmapContextCreate(_pRgbImageBuf, _imageWidth, _imageHeight, 8, _bytesPerRow, self.colorSpace,
                                                 kCGBitmapByteOrder32Little | kCGImageAlphaNoneSkipLast);
    
    self.pixelNum = _imageWidth * _imageHeight;// 遍历像素总数
    self.conx = context;
//    CGContextDrawImage(self.conx, CGRectMake(0, 0, imageWidth, imageHeight), self.image.CGImage);
    CGContextSetRGBFillColor(self.conx, 1, 1, 1, 1);
    CGContextFillRect(self.conx, CGRectMake(0, 0, _imageWidth, _imageHeight));
    CGContextStrokePath(self.conx);
    self.dataProvider = CGDataProviderCreateWithData(NULL, _pRgbImageBuf, _bytesPerRow * _imageHeight, providerReleaseData);
    self.fps = fps;
    
    NSBundle *bundle = [NSBundle bundleWithPath:[[NSBundle mainBundle] pathForResource:@"source" ofType:@"bundle"]];
    self.brushMaskImageArray = [NSMutableArray array];
    for (int i=1; i<=BRUSH_COUNTS; ++i) {
        [self.brushMaskImageArray addObject:[UIImage imageNamed:[NSString stringWithFormat:@"mask%d.jpg",i] inBundle:bundle compatibleWithTraitCollection:nil]];
    }
    self.isPrepare = YES;
    
}


-(void)removeFromSuperview{
    //当对象呗移除时，清理内存，释放上下文。
    NSLog(@"removeFromSuperview");
    [self clean];
    [super removeFromSuperview];
}



-(void)clean{
    //重新绘图或退出时，释放上下文
    NSLog(@"clean");
    self.bDrawing = NO;
    if (self.conx) {
        CGDataProviderRelease(self.dataProvider);
        CGContextRelease(self.conx);
        CGColorSpaceRelease(self.colorSpace);
        self.conx = nil;
        self.colorSpace = nil;
        self.dataProvider = nil;
        self.fps = 0;
    }
    [self.brushMaskImageArray removeAllObjects];
    self.brushMaskImageArray = nil;

}
-(void)dealloc{
    NSLog(@"dealloc");
}




void providerReleaseData (void *info, const void *data, size_t size){
    //重新绘图或退出时，释放pRgbImageBuf
    free((void*)data);
}




//===================================================

+ (UIImage *)imageFromCVMat:(cv::Mat)cvMat {
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

- (cv::Mat)cvMatRepresentationColor
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(self.image.CGImage);
    CGFloat cols = self.image.size.width;
    CGFloat rows = self.image.size.height;
    
    cv::Mat color(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(color.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    color.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), self.image.CGImage);
    CGContextRelease(contextRef);
    
    return color;
}


- (cv::Mat)cvMatRepresentationGray
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(self.image.CGImage);
    int cols = self.image.size.width;
    int rows = self.image.size.height;
    
    Mat gray(rows, cols, CV_8UC1);
    
    NSLog(@"cols %d rows %d step %zu", cols, rows, gray.step[0]);
    CGContextRef contextRef = CGBitmapContextCreate(gray.data,                 // Pointer to data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    gray.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), self.image.CGImage);
    CGContextRelease(contextRef);
    
    return gray;
}

+ (cv::Mat)oilPainting:(cv::Mat)cvMat{
    //油画,失败
    cv::Mat tmpCvMat,fan,guass;
    cv::Mat resualt;
    cv::Mat tmp;
    
    int basicSize = 4;
    int grayLevelSize = 8;
    int gap = 12;
    
    
    if(!cvMat.empty()){
        resualt = Mat::zeros(cvMat.size(),cvMat.type());
        cv::cvtColor(cvMat,tmpCvMat,CV_RGB2GRAY);
        for(int y=basicSize;y<cvMat.rows-basicSize*2;y+=gap){
            for(int x=basicSize;x<cvMat.cols-basicSize*2;x+=gap){
                //灰度等级统计
                std::vector<int> grayLevel(grayLevelSize);
                Vec4i graySum(0,0,0,255);
                //对小区域进行遍历统计
                for(int m=-basicSize; m<=basicSize; ++m){
                    for(int n=-basicSize; n<=basicSize; ++n){
                        char tmp = tmpCvMat.at<char>(y+m, x+n);
                        int pixlv = int(int(tmp) / (256 / grayLevelSize));//判断像素等级
                        grayLevel[pixlv] += 1;//计算对应灰度等级个数
                    }}
                //找出最高频灰度等级及其索引
                std::vector<int>::iterator biggest = std::max_element(std::begin(grayLevel), std::end(grayLevel));
                int mostLevel = *biggest;
                long mostLevelIndex = std::distance(std::begin(grayLevel), biggest);
                //计算最高频等级内的所有灰度值的均值
                for(int m=-basicSize; m<=basicSize; ++m){
                    for(int n=-basicSize; n<=basicSize; ++n){
                        char tmp = tmpCvMat.at<char>(y+m, x+n);
                        int pixlv = int(int(tmp)  / (256 / grayLevelSize));//判断像素等级
                        if(pixlv == mostLevelIndex){
                            graySum += cvMat.at<Vec4b>(y+m, x+n);
                        }}}
                Vec4b rgb;
                if(mostLevel!=0){
                    //                    printf("%d,%d,%d \n",graySum[0],int(graySum[1]),int(graySum[2]));
                    //                    printf("%d,%d,%d \n",int(graySum[0]/mostLevel),int(graySum[1]/mostLevel),int(graySum[2]/mostLevel));
                    rgb = Vec4b(int(graySum[0]/mostLevel),int(graySum[1]/mostLevel),int(graySum[2]/mostLevel),0);
                }else{
                    printf("NNNN %d,%d,%d \n",graySum[0],int(graySum[1]),int(graySum[2]));
                    rgb = Vec4b(0,0,0,0);
                    continue;
                }
                //写入目标像素
                for(int m=0; m<gap; ++m){
                    for(int n=0; n<gap; ++n){
                        resualt.at<Vec4b>(y+m, x+n) = rgb;
                    }}
                
            }
        }
        
    }
    
    resualt = resualt;
    return resualt;
}


-(instancetype)initWithFrame:(CGRect)frame{
    self = [super initWithFrame:frame];
//    self.contentMode = UIViewContentModeScaleAspectFill;
    self.contentMode = UIViewContentModeScaleAspectFit;
    self.bDrawing = NO;
    self.sketchDurationMultiplying = 1.f;
    self.groundingDurationMultiplying = 1.f;
    self.lineDurationMultiplying = 1.f;
    self.originDurationMultiplying = 1.f;
    return self;
}

+ (cv::Mat)paintByGuass:(cv::Mat)cvMat{
    //灰度图取反
    cv::Mat tmpCvMat,fan,guass;
    cv::Mat resualt;
    if(!cvMat.empty()){
        cv::cvtColor(cvMat,tmpCvMat,CV_RGB2GRAY);
        fan = Mat::zeros(tmpCvMat.size(),tmpCvMat.type());
        guass = Mat::zeros(tmpCvMat.size(),tmpCvMat.type());
        resualt = Mat::zeros(cvMat.size(),cvMat.type());
        for(int y=0;y<fan.rows;++y)
            for(int x=0;x<fan.cols;++x)
                fan.at<uchar>(y,x)=255-tmpCvMat.at<uchar>(y,x);
        
        cv::GaussianBlur(fan,guass,cv::Size(11,11),0,0);
        
        for(int y=0;y<guass.rows;++y){
            for(int x=0;x<guass.cols;++x){
                double b =double(guass.at<uchar>(y,x));
                double a =double(tmpCvMat.at<uchar>(y,x));
                int tmp = int(a+a*b/(256-b));
                tmp = tmp<255?tmp:255;
                Vec4b tp(tmp,tmp,tmp,255);
                resualt.at<Vec4b>(y,x)=tp;
            }
        }
    }
    //    cv::print(resualt);
    resualt = resualt;
    return resualt;
}



+ (cv::Mat)lineDraftByMorphology:(cv::Mat)cvMat{
    //线稿
    cv::Mat tmpCvMat,fan,guass;
    cv::Mat resualt;
    cv::Mat tmp;
    
    if(!cvMat.empty()){
        cv::cvtColor(cvMat,tmpCvMat,CV_RGB2GRAY);
        int kernelWidth = (int)(MIN(cvMat.cols, cvMat.rows) * 0.01);
        cv::Size kernelSize(kernelWidth, kernelWidth);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, kernelSize);
        cv::morphologyEx(tmpCvMat, tmpCvMat, MORPH_GRADIENT, kernel);
        cv::threshold(tmpCvMat,resualt,80,80,CV_THRESH_TRUNC);
        cv::bitwise_not(resualt,resualt);
    }
    
    resualt = resualt-tmpCvMat;
    return resualt;
}

+ (cv::Mat)keypoint:(cv::Mat)cvMat{
    //关键点
    cv::Mat tmpCvMat;
    cv::Mat resualt;
    cv::Mat keyPointImage1,keyPointImage2;
    if(!cvMat.empty()){
        cv::cvtColor(cvMat,tmpCvMat,CV_RGB2GRAY);
        vector<cv::KeyPoint>detectKeyPoint;
        //        Ptr<GFTTDetector> detector = cv::GFTTDetector::create();
        //        detector->detect(tmpCvMat,detectKeyPoint);
        //        Ptr<SimpleBlobDetector> detector = cv::SimpleBlobDetector::create();
        //        detector->detect(tmpCvMat,detectKeyPoint);
        
        cv::FAST(tmpCvMat, detectKeyPoint, 20);
        //        cv::AGAST(tmpCvMat, detectKeyPoint, 20);
        drawKeypoints(tmpCvMat,detectKeyPoint,keyPointImage1,Scalar(0,0,255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(tmpCvMat,detectKeyPoint,keyPointImage2,Scalar(0,0,255),DrawMatchesFlags::DEFAULT);
    }
    
    resualt = keyPointImage1;
    return resualt;
}

+ (cv::Mat)focusBySobelTenenGrad:(cv::Mat)cvMat{
    cv::Mat tmpCvMat;
    cv::Mat resualt;
    
    int ROISize = 4;
    
    if(!cvMat.empty()){
        cv::cvtColor(cvMat,tmpCvMat,CV_RGB2GRAY);
        cv::equalizeHist(tmpCvMat, tmpCvMat);
        cv::medianBlur(tmpCvMat, tmpCvMat, 1);
        Mat imageSobel;
        Sobel(tmpCvMat, imageSobel, CV_16U, 1, 1);
        double meanValue = 0.0;
        double maxValue = meanValue;
        int movCols = cvMat.cols / 20;
        int movRows = cvMat.rows / 20;
        cv::Rect select;
        for (int j = 0; (cvMat.rows / ROISize + j*movRows) <= (cvMat.rows); j++){
            for (int i = 0; (cvMat.cols / ROISize + i*movCols) <= (cvMat.cols); i++){
                cv::Mat ROI = imageSobel(Range(0 + j*movRows, cvMat.rows / ROISize + j*movRows), Range(0 + i*movCols, cvMat.cols / ROISize + i*movCols));
                
                if (double(cv::mean(ROI)[0]) > maxValue){
                    maxValue = cv::mean(ROI)[0];
                    
                    select.x = i*movCols;
                    select.y = j*movRows;
                    select.width = cvMat.cols / ROISize;
                    select.height = cvMat.rows / ROISize;
                }
            }
        }
        cv::rectangle(cvMat, select, Scalar(0, 255, 0), 2);
    }
    resualt = cvMat;
    return resualt;
    
}

+ (cv::Mat)splitFrontAndBackgroundByInRang:(cv::Mat)cvMat{
    cv::Mat masker;
    cv::Mat pieces;
    if(!cvMat.empty()){
        cv::Scalar meanColor;
        cv::Scalar stdDevColor;
        cv::meanStdDev(cvMat, meanColor, stdDevColor);
        // Create a mask based on a range around the mean color.
        cv::Scalar halfRange = 0.8 * stdDevColor;
        cv::Scalar lowerBound = meanColor - halfRange;
        cv::Scalar upperBound = meanColor + halfRange;
        cv::inRange(cvMat, lowerBound, upperBound, masker);
        
        // Erode the mask to merge neighboring blobs.
        int kernelWidth = (int)(MIN(cvMat.cols, cvMat.rows) * 0.01);
        if(kernelWidth > 0) {
            cv::Size kernelSize(kernelWidth, kernelWidth);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, kernelSize);
            cv::erode(masker, masker, kernel, cv::Point(-1, -1), 1);
        }
        
        //检索模式：
        //cv::RETR_EXTERNAL表示只检测外轮廓
        //cv::RETR_LIST检测的轮廓不建立等级关系
        //cv::RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
        //cv::RETR_TREE建立一个等级树结构的轮廓。
        //轮廓的近似方法
        //cv::CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
        //cv::CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(masker, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        
        for(int index=0;index<contours.size();++index){
            vector<cv::Point> curve,approx;
            cv::approxPolyDP(contours[index], approx, 5, true);
            std::vector<std::vector<cv::Point>> ttt;
            ttt.push_back(approx);
            cv::drawContours(cvMat, ttt, -1, cv::Scalar(arc4random()%250,arc4random()%250,arc4random()%250),3);
            NSLog(@"%d:%d,%d,%d,%d",index,hierarchy[index][0],hierarchy[index][1],hierarchy[index][2],hierarchy[index][3]);
        }
        
        NSLog(@"contours.size,%lu",contours.size());
        //        cv::drawContours(cvMat, contours, -1, cv::Scalar(255,0,0),3);
        //        cv::contourArea()
        pieces = cvMat;
    }
    
    return pieces;
}

-(void)daubingAvgColorGrounding:(std::vector<std::vector<cv::Point>>*)areasVec orderlyPointsVec:(std::vector<std::vector<cv::Point>> *)orderlyPointsVec groundingMat:(cv::Mat*) groundingMat brushmaskMat:(cv::Mat*)brushmaskMat frameCount:(int)frameCount totalAreaSize:(double)totalAreaSize {
    std::vector<std::vector<cv::Point>> &areas = *areasVec;
    std::vector<std::vector<cv::Point>> &points = *orderlyPointsVec;
    cv::Mat &grounding = *groundingMat;
    cv::Mat &brushmask = *brushmaskMat;
    
    std::vector<std::vector<cv::Point>> tmpVecsInVec;//提高效率，避免重复创建
    tmpVecsInVec.push_back(areas[0]);
    cv::Mat roi = Mat::zeros(grounding.size(),CV_8U);//提高效率，避免重复创建
    cv::Mat inverseROI = cv::Mat(grounding.size(),CV_8U,cv::Scalar::all(255));//提高效率，避免重复创建
    cv::Mat backgroundROI = cv::Mat(grounding.size(),CV_8UC3,cv::Scalar(255,0,255));//提高效率，避免重复创建
    
    //创建绘制顺序的索引，从大区域开始。
    std::vector<cv::Vec2d> sizeIndexes;
    for(int areaIndex=0; areaIndex<areas.size(); ++areaIndex){
        sizeIndexes.push_back(cv::Vec2d(areaIndex,cv::contourArea(areas[areaIndex])));
    }
    std::sort(sizeIndexes.begin(), sizeIndexes.end(), comp);
    
    int accumulative = 1;
    int orderIndex = -1;
    for(cv::Vec2d indexVec:sizeIndexes){
        int areaIndex =indexVec[0];
        double areaSize = indexVec[1];
        
        orderIndex++;
        
        if(areaSize==0){
            //            continue;
            break;
        }
        
        
        //控制绘制速度
        std::vector<cv::Point> pointsInArea = points[areaIndex];
        //本次绘制区域中的引导点数量，并不是真正的绘制点，因为笔刷效果一次画的数量为笔刷宽度的2倍。
        int pointsCountInArea = int(pointsInArea.size());
        //通过本区域的面积占总体面积的比例，计算本区域应该承担整个上色动画的多少帧画面
        double areaRate = areaSize/totalAreaSize;
        double framesInThisArea = frameCount * areaRate;
        framesInThisArea = framesInThisArea<1?1:framesInThisArea;
        int pointsInFrame = int(pointsCountInArea/framesInThisArea);//一帧内应该包含多少绘图引导点。既笔刷的中心点。
        
        //        printf("[%d]:%d, %.1f, %f\n",orderIndex,areaIndex,areaSize,areaRate);
        if(orderIndex>800 && areaSize<100 && areaRate<.00005){
            break;//大量小块不再处理直接跳过。
        }
        
        Mat dst;//真正的绘制目标区域；
        
        //提取绘制区域内容到dst
        tmpVecsInVec[0] = areas[areaIndex];
        
        //设置拷贝上色区
        cv::drawContours(roi,tmpVecsInVec,0,Scalar::all(255),-1);
        grounding.copyTo(dst,roi);
        cv::Scalar avgChannels = cv::mean(dst,roi);
        cv::drawContours(roi,tmpVecsInVec,0,Scalar::all(0),-1);
        
        //设置背景色
        cv::drawContours(inverseROI,tmpVecsInVec,0,Scalar::all(0),-1);
        backgroundROI.copyTo(dst,inverseROI);
        cv::drawContours(inverseROI,tmpVecsInVec,0,Scalar::all(255),-1);
        
        
        int brushWidth = kk_calculateBrushWidth(tmpVecsInVec[0]);
        cv::Vec4i drawRect = kk_getMinRectangleRect(tmpVecsInVec[0]);
        
        std::vector<cv::Point> truelyBrushCompensatePoints;
        std::vector<cv::Vec3b> truelyBrushCompensateColors;
        
        
        for (int i=0; i<pointsInArea.size(); ++i) {
            cv::Point startP = pointsInArea[i];
            for (int j=0; j<brushWidth; j++) {//如果笔刷大于1，将横向读取产生批量宽度笔刷效果
                //从引导点往左一个笔刷宽度开始，画2倍笔刷。多出的部分用边界控制，实际上并未产生多少浪费。如果1倍笔刷，在边界上会存在遗漏，形成锯齿。
                cv::Point p = cv::Point(startP.x-j,startP.y);
                if (p.x > drawRect[2]+2)continue;//控制越界，减少性能浪费
                if (p.x < drawRect[0]-1)continue;//控制越界，减少性能浪费
                Vec3b pixelVec3b = dst.at<Vec3b>(p);
                //赤红背景不进行绘制，因为区域实际上是多边形，但是绘制为了方便和效率，实际上取的是矩形，多余部分不取
                if(pixelVec3b[0]!=255&&pixelVec3b[1]!=0&&pixelVec3b[2]!=255){
                    if (areaRate>0.3) {//触发真实笔刷
                        uchar mask = brushmask.at<uchar>(p);
                        if (mask > 5) {
                            //掩版高亮部分绘制
                            Vec3b pixel = Vec3b(char(int(avgChannels.val[0])),char(int(avgChannels.val[1])),char(int(avgChannels.val[2])));
                            kk_updateImageBuffByPixelVec3b(&p, &pixel, _pRgbImageBuf, _imageWidth, AVG);
                        }else{
                            //掩版俺部绘制存储，后补绘制。
                            Vec3b pixel = Vec3b(char(int(avgChannels.val[0])),char(int(avgChannels.val[1])),char(int(avgChannels.val[2])));
                            truelyBrushCompensatePoints.push_back(p);
                            truelyBrushCompensateColors.push_back(pixel);
                        }
                    }else{
                        //为了节省效率，避免mask在不需要的时候被读取，重复一遍一下逻辑
                        Vec3b pixel = Vec3b(char(int(avgChannels.val[0])),char(int(avgChannels.val[1])),char(int(avgChannels.val[2])));
                        kk_updateImageBuffByPixelVec3b(&p, &pixel, _pRgbImageBuf, _imageWidth, AVG);
                    }
                }
                accumulative++;
            }
            
            if(i!=0 && (i % pointsInFrame) == 0){
                [self updateImage];
                [NSThread sleepForTimeInterval:1.f/self.fps];
            }else if (i==pointsInArea.size()-1){
                if (areas.size()>600 && areaRate<.00005) {//控制背景过于复查的情况，小细节就不一直占用帧数了。
                    if ((accumulative % pointsInFrame/2) == 0) {
                        accumulative = 1;
                        [self updateImage];//[NSThread sleepForTimeInterval:(1.f/self.fps)*(areaRate+0.2)];//防止最后一些非常小的区域占用太多帧数
                    }else{
                        accumulative++;
                    }
                }else{
                    [self updateImage];
                    [NSThread sleepForTimeInterval:(1.f/self.fps)*(areaRate+0.2)];//防止最后一些非常小的区域占用太多帧数
                }
            }
        }
        //
        //真实笔刷会有遗漏点，当绘图过半时进行补偿
        if (areaRate>0.3 && truelyBrushCompensatePoints.size()>0) {
            for (int comIdx=0; comIdx<truelyBrushCompensatePoints.size(); ++comIdx) {
                cv::Point comP = truelyBrushCompensatePoints[comIdx];
                Vec3b comPixel = truelyBrushCompensateColors[comIdx];
                kk_updateImageBuffByPixelVec3b(&comP, &comPixel, _pRgbImageBuf, _imageWidth, AVG);
                if( (comIdx!=0 && (comIdx % (pointsInFrame*brushWidth))==0) || comIdx==pointsInArea.size()-1){
                    [self updateImage];
                    [NSThread sleepForTimeInterval:1.f/self.fps];
                }
            }
        }
    }
    [self updateImage];//函数结束前再更新一下避免遗漏。
}


@end

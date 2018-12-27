//
//  UIPaintingImageView.h
//  PaintView
//
//  Created by kk on 2018/10/25.
//  Copyright © 2018年 kk. All rights reserved.
//


#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <UIKit/UIKit.h>
#import <QuartzCore/QuartzCore.h>
#import <CoreGraphics/CGContext.h>

#import <opencv2/imgproc/types_c.h>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/core/hal/interface.h>
#import <opencv2/core/operations.hpp>
#import <opencv2/objdetect/objdetect.hpp>


using namespace cv;
using namespace std;

//笔刷的蒙版数量，与source.bundle中的mask%d.jpg有关。
#define BRUSH_COUNTS 3

//绘制草图的基础持续时间。
#define DURATION_SKETCH 4
//大背景上色的基础持续时间。
#define DURATION_AVG_GROUNDING 3
//水彩上色的基础持续时间。
#define DURATION_GROUNDING 2
//绘制细节的基础持续时间。
#define DURATION_ORIGIN 2
//最后收尾、补充原图细节的基础持续时间。
#define DURATION_FINAL 2

//原图的最小边宽，大于此限制的图片将被重新等比采样至限制边宽。
#define LIMIT_ORIGINSIZE_F 1500.f

//笔刷宽度相对区块宽度的倍率
#define RATE_BRUSHWIDTH_GROUNDING 0.1
//笔刷生效宽度，此宽度以下以像素为单位绘制
#define LIMIT_AREASIZE_BRUSHWIDTH 100
//触发真实笔刷特效的面积占比。大于此面积的绘图区才触发真实笔刷特效
#define LIMIT_TRUE_BRUSH_AREARATE 0.3


typedef NS_ENUM(int) {
    NORMAL,
    AVG,
    GROUNDING1,
    GROUNDING2,
    ORIGINIMAGE1,
    ORIGINIMAGE2,
    FINAL
}BUSHINGTYPY;

@protocol paintingDelegate <NSObject>
-(void)drawEnding;
@end


@interface UIPaintingImageView: UIImageView


-(void)clean;

- (void)startDrawing;
-(void)prepareForPainting:(UIImage*)img fps:(uint)fps sketchTimeMulti:(float)stm groundingTimeMulti:(float)gtm lineTimeMulti:(float)ltm originTimeMulti:(float)otm;

@property (nonatomic, weak) id<paintingDelegate> delegate;

@property (nonatomic, strong) dispatch_queue_t serialQueue;
@property (nonatomic, strong) dispatch_group_t group;

@property (nonatomic, assign) CGContextRef conx;
@property (nonatomic, assign) CGColorSpaceRef colorSpace;
@property (nonatomic, assign) CGDataProviderRef dataProvider;

@property (nonatomic, assign) BOOL bDrawing;
@property (nonatomic, assign) uint fps;

@property (nonatomic, assign) int pixelNum ;
@property (nonatomic, assign) uint32_t* pRgbImageBuf;
@property (nonatomic, assign) int imageWidth;
@property (nonatomic, assign) int imageHeight;
@property (nonatomic, assign) size_t bytesPerRow;

@property (nonatomic, strong) NSMutableArray *brushMaskImageArray;
@property (nonatomic, assign) float sketchDurationMultiplying;
@property (nonatomic, assign) float groundingDurationMultiplying;
@property (nonatomic, assign) float lineDurationMultiplying;
@property (nonatomic, assign) float originDurationMultiplying;
@property (nonatomic, assign) bool isPrepare;
@end

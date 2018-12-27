//
//  demo_ViewController.h
//  PaintView
//
//  Created by kk on 2018/11/18.
//  Copyright © 2018年 kk. All rights reserved.
//

#import "UIPaintingImageView.h"
#import <UIKit/UIKit.h>
#import "demo_UIDragBar.h"
#import <AVFoundation/AVCaptureDevice.h>
#import <AVFoundation/AVMediaFormat.h>
#import <Photos/Photos.h>


#define THRESHOLD_DIC_FILE [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject  stringByAppendingPathComponent:@"threshold.plist"]
#define DRAGBAR_HEIGHT 300

#define THRESHOLD_SKETCH @"sketch"
#define THRESHOLD_GROUNDING @"grounding"
#define THRESHOLD_LINE @"line"
#define THRESHOLD_ORIGIN @"origin"

#define AV @"camera"
#define PH @"photo"


@interface demo_ViewController : UIViewController

@property (strong, nonatomic) IBOutlet UIButton *loadimgBtn;
- (IBAction)loadimgBtnToPressed:(id)sender;
@property (strong, nonatomic) IBOutlet UIImageView *imgView;
@property (strong, nonatomic) UIPaintingImageView *paintingView;
@property (strong, nonatomic) demo_UIDragBar *dragBar;


@property (strong, nonatomic) IBOutlet UIButton *sketchBtn;
- (IBAction)sketchBtnToPressed:(id)sender;
@property (strong, nonatomic) IBOutlet UIButton *goundingBtn;
- (IBAction)goundingBtnToPressed:(id)sender;
@property (strong, nonatomic) IBOutlet UIButton *lineBtn;
- (IBAction)lineBtnToPressed:(id)sender;
@property (strong, nonatomic) IBOutlet UIButton *originBtn;
- (IBAction)originBtnToPressed:(id)sender;
@property (strong, nonatomic) IBOutlet UIButton *playBtn;
- (IBAction)playBtnToPressed:(id)sender;

@property (strong, nonatomic) IBOutlet UIView *debugView;

@property (strong, nonatomic) NSMutableDictionary *thresholdsDic;
@property (assign, nonatomic) bool thresholdsHadChanged;


@end


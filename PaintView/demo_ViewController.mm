//
//  demo_ViewController.m
//  PaintView
//
//  Created by kk on 2018/11/18.
//  Copyright © 2018年 kk. All rights reserved.
//

#import "demo_ViewController.h"



@interface demo_ViewController ()<UIImagePickerControllerDelegate, UINavigationControllerDelegate,paintingDelegate,UIDragBarProtocol>
@end

@implementation demo_ViewController


- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.thresholdsHadChanged = NO;
    self.thresholdsDic = [NSMutableDictionary dictionaryWithContentsOfFile:THRESHOLD_DIC_FILE];
    if (self.thresholdsDic == nil) {
        NSLog(@"null");
        self.thresholdsDic = [NSMutableDictionary dictionaryWithObjects:@[@1.f,@1.f,@1.f,@1.f] forKeys:@[THRESHOLD_SKETCH,THRESHOLD_GROUNDING,THRESHOLD_LINE,THRESHOLD_ORIGIN]];
        [_thresholdsDic writeToFile:THRESHOLD_DIC_FILE atomically:YES];
    }

    [self initDebugBar];
    
    
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(orientChange:) name:UIDeviceOrientationDidChangeNotification object:nil];

    
}

-(void)initDebugBar{
    UIColor *col;
    const CGFloat *components;
    col = [demo_UIDragBar calculateCurColor:[_thresholdsDic[THRESHOLD_SKETCH]floatValue] height:DRAGBAR_HEIGHT];
    [_sketchBtn setBackgroundColor:col];
    components = CGColorGetComponents(col.CGColor);
    [_sketchBtn setTitleColor:[UIColor colorWithRed:1-components[0] green:1-components[1] blue:1-components[2] alpha:1] forState:UIControlStateNormal];
    [_sketchBtn setTitle:[NSString stringWithFormat:@"X%.1f%@",[_thresholdsDic[THRESHOLD_SKETCH]floatValue],[_sketchBtn.titleLabel.text substringFromIndex:4]] forState:UIControlStateNormal];

    col = [demo_UIDragBar calculateCurColor:[_thresholdsDic[THRESHOLD_GROUNDING]floatValue] height:DRAGBAR_HEIGHT];
    [_goundingBtn setBackgroundColor:col];
    components = CGColorGetComponents(col.CGColor);
    [_goundingBtn setTitleColor:[UIColor colorWithRed:1-components[0] green:1-components[1] blue:1-components[2] alpha:1] forState:UIControlStateNormal];
    [_goundingBtn setTitle:[NSString stringWithFormat:@"X%.1f%@",[_thresholdsDic[THRESHOLD_GROUNDING]floatValue],[_goundingBtn.titleLabel.text substringFromIndex:4]] forState:UIControlStateNormal];

    col = [demo_UIDragBar calculateCurColor:[_thresholdsDic[THRESHOLD_LINE]floatValue] height:DRAGBAR_HEIGHT];
    [_lineBtn setBackgroundColor:col];;
    components = CGColorGetComponents(col.CGColor);
    [_lineBtn setTitleColor:[UIColor colorWithRed:1-components[0] green:1-components[1] blue:1-components[2] alpha:1] forState:UIControlStateNormal];
    [_lineBtn setTitle:[NSString stringWithFormat:@"X%.1f%@",[_thresholdsDic[THRESHOLD_LINE]floatValue],[_lineBtn.titleLabel.text substringFromIndex:4]] forState:UIControlStateNormal];

    col = [demo_UIDragBar calculateCurColor:[_thresholdsDic[THRESHOLD_ORIGIN]floatValue] height:DRAGBAR_HEIGHT];
    [_originBtn setBackgroundColor:col];
    components = CGColorGetComponents(col.CGColor);
    [_originBtn setTitleColor:[UIColor colorWithRed:1-components[0] green:1-components[1] blue:1-components[2] alpha:1] forState:UIControlStateNormal];
    [_originBtn setTitle:[NSString stringWithFormat:@"X%.1f%@",[_thresholdsDic[THRESHOLD_ORIGIN]floatValue],[_originBtn.titleLabel.text substringFromIndex:4]] forState:UIControlStateNormal];

}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (IBAction)loadimgBtnToPressed:(id)sender{
    NSLog(@"load");
    [self boot];
    [self headClick];
}
- (void)dismiss:(UIAlertController *)alert{
    [alert dismissViewControllerAnimated:YES completion:nil];
}


-(void)drawEnding{
    UIAlertController *actionSheet = [UIAlertController alertControllerWithTitle:@"绘制完成" message:nil preferredStyle:UIAlertControllerStyleActionSheet];
    if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPad){
        actionSheet.popoverPresentationController.sourceView = self.view;
        actionSheet.popoverPresentationController.sourceRect = CGRectMake(self.view.frame.size.width/4,2*(self.view.frame.size.height/3),self.view.frame.size.width/2,1.0);
    }
    [self presentViewController:actionSheet animated:YES completion:nil];
    [self performSelector:@selector(dismiss:) withObject:actionSheet afterDelay:1.8];
    [self.playBtn setTitle:@"play" forState:UIControlStateNormal];
}


- (void)updateThreshold:(NSString*)name threshold:(float)threshold color:(UIColor*)color sender:(id)sender {
    if (sender == nil || name == nil ) return;
    UIButton *btn = (UIButton *)sender;
    [btn setBackgroundColor:color];
    const CGFloat *components = CGColorGetComponents(color.CGColor);
    [btn setTitleColor:[UIColor colorWithRed:1-components[0] green:1-components[1] blue:1-components[2] alpha:1] forState:UIControlStateNormal];
    [btn setTitle:[NSString stringWithFormat:@"X%.1f%@",threshold,[btn.titleLabel.text substringFromIndex:4]] forState:UIControlStateNormal];

    if ([_thresholdsDic[name]floatValue] != threshold) {
        _thresholdsDic[name] = [NSNumber numberWithFloat:threshold];
        self.thresholdsHadChanged = YES;
        NSLog(@"changed");
    }
}
- (void)dragBarsuicide{
    self.dragBar = nil;
}

-(void)showOrHideDragBar:(id)sender barName:(NSString*)barName threshold:(float)threshold{
    UIButton *btn = sender;
    if (self.dragBar) {
        bool onlyRemove = (btn == _dragBar.pFather);
        [_dragBar hide];
        self.dragBar = nil;
        if (onlyRemove) {
            return;
        }
    }
    int w = btn.frame.size.width*0.8;
    int h = DRAGBAR_HEIGHT;
    self.dragBar = [[demo_UIDragBar alloc]initWithFrame:CGRectMake(btn.frame.origin.x+(btn.frame.size.width-w)/2, btn.frame.origin.y-h, w, h)];
    [self.dragBar setDelegate:self];
    self.dragBar.pFather = btn;
    [self.dragBar setName:barName];
    [self.view addSubview:self.dragBar];
    [self.view bringSubviewToFront:btn];
    [_dragBar setNumber:threshold];
    [_dragBar show];
}


- (IBAction)sketchBtnToPressed:(id)sender{
    [self showOrHideDragBar:sender barName:THRESHOLD_SKETCH threshold:[_thresholdsDic[THRESHOLD_SKETCH]floatValue]];
}

- (IBAction)goundingBtnToPressed:(id)sender{
    [self showOrHideDragBar:sender barName:THRESHOLD_GROUNDING threshold:[_thresholdsDic[THRESHOLD_GROUNDING]floatValue]];
}

- (IBAction)lineBtnToPressed:(id)sender{
    [self showOrHideDragBar:sender barName:THRESHOLD_LINE threshold:[_thresholdsDic[THRESHOLD_LINE]floatValue]];
}

- (IBAction)originBtnToPressed:(id)sender{
    [self showOrHideDragBar:sender barName:THRESHOLD_ORIGIN threshold:[_thresholdsDic[THRESHOLD_ORIGIN]floatValue]];
}

- (IBAction)playBtnToPressed:(id)sender{
//    NSData * imgData =[NSData dataWithContentsOfFile:@"/Users/kk/Desktop/7.jpg"];
//    self.imgView.image = [UIImage imageWithData:imgData];
    
    if (self.imgView.image == nil){
        [self esayAlert:@"提示" message:@"请先加载图片"];
        return;
    }
    if (([_thresholdsDic[THRESHOLD_ORIGIN]floatValue]+[_thresholdsDic[THRESHOLD_LINE]floatValue]+[_thresholdsDic[THRESHOLD_GROUNDING]floatValue]+[_thresholdsDic[THRESHOLD_SKETCH]floatValue])==.0f){
        [self esayAlert:@"提示" message:@"需要至少有一项播放时间因子不为0。"];
        return;
    }
    if (_thresholdsHadChanged) {
        [_thresholdsDic writeToFile:THRESHOLD_DIC_FILE atomically:YES];
        self.thresholdsHadChanged = NO;
    }
    if (self.dragBar) {
        [_dragBar hide];
        self.dragBar = nil;
    }
    [self.playBtn setTitle:@"rePlay" forState:UIControlStateNormal];
    
    
//    NSBundle *bundle = [NSBundle bundleWithPath:[[NSBundle mainBundle] pathForResource:@"source" ofType:@"bundle"]];
//    NSData * imgData =[NSData dataWithContentsOfFile:[bundle pathForResource:@"35" ofType:@"jpg"]];
    if (self.paintingView) {
        [self.paintingView removeFromSuperview];
        self.paintingView = nil;
    }
    CGRect mainScreen = [[UIScreen mainScreen] bounds];
    self.paintingView = [[UIPaintingImageView alloc]initWithFrame:CGRectMake(0, 0, mainScreen.size.width, mainScreen.size.height-self.debugView.frame.size.height)];
    [self.paintingView prepareForPainting:self.imgView.image fps:30
                          sketchTimeMulti:[_thresholdsDic[THRESHOLD_SKETCH]floatValue]
                       groundingTimeMulti:[_thresholdsDic[THRESHOLD_GROUNDING]floatValue]
                            lineTimeMulti:[_thresholdsDic[THRESHOLD_LINE]floatValue]
                          originTimeMulti:[_thresholdsDic[THRESHOLD_ORIGIN]floatValue]];
    
    self.paintingView.delegate = self;
    
    [self.view addSubview:self.paintingView];
    
    [self.paintingView startDrawing];

}


- (void)orientChange:(NSNotification *)noti {
    if (self.dragBar) {
        [_dragBar hide];
        self.dragBar = nil;
    }
    if(self.paintingView!=nil){
        CGRect mainScreen = [[UIScreen mainScreen] bounds];
        [self.paintingView setFrame:CGRectMake(0, 0, mainScreen.size.width, mainScreen.size.height-self.debugView.frame.size.height)];
    }

    UIDeviceOrientation  orient = [UIDevice currentDevice].orientation;
    switch (orient) {
        case UIDeviceOrientationPortrait:
            break;
        case UIDeviceOrientationLandscapeLeft:
            break;
        case UIDeviceOrientationPortraitUpsideDown:
            break;
        case UIDeviceOrientationLandscapeRight:
            break;
        default:
            break;
    }
}

#pragma mark -相机相关事件-
- (void)boot{
    AVAuthorizationStatus avStatus = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
    
    //  avStatus 的4中类型
    /*
     AVAuthorizationStatusNotDetermined  // 初次调用
     AVAuthorizationStatusRestricted //  禁用
     AVAuthorizationStatusDenied //
     AVAuthorizationStatusAuthorized // 开通权限
     */
    
    // 用户开放相机权限后 判断相机是否可用
    BOOL useable = [UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypeCamera];
    
    if (!TARGET_IPHONE_SIMULATOR) {
        //模拟器不支持
        PHAuthorizationStatus phStatus = [PHPhotoLibrary authorizationStatus];
    }
    // 同样 phStatus 有4中类型
    /*
     PHAuthorizationStatusNotDetermined = 0
     PHAuthorizationStatusRestricted
     PHAuthorizationStatusDenied
     PHAuthorizationStatusAuthorized
     */
    
    // 判断相册权限
    [UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypePhotoLibrary];
}


//用户关闭iOS拍照/相册权限,引导用户打开拍照/相册权限
- (void)guideUserOpenAuth{
    UIAlertController *alertC = [UIAlertController alertControllerWithTitle:@"温馨提示" message:@"请打开访问权限" preferredStyle:(UIAlertControllerStyleAlert)];
    if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPad){
        alertC.popoverPresentationController.sourceView = self.view;
        alertC.popoverPresentationController.sourceRect = CGRectMake(self.view.frame.size.width/4,2*(self.view.frame.size.height/3),self.view.frame.size.width/2,1.0);
    }
    
    

    UIAlertAction *alertA = [UIAlertAction actionWithTitle:@"确定" style:(UIAlertActionStyleDefault) handler:nil];
    UIAlertAction *act = [UIAlertAction actionWithTitle:@"去设置" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
        // 引导用户设置
        NSURL *url = [NSURL URLWithString:UIApplicationOpenSettingsURLString];
        
        if ([[UIApplication sharedApplication] canOpenURL:url]) {
            
            [[UIApplication sharedApplication] openURL:url options:@{} completionHandler:nil];
        }
    }];
    [alertC addAction:alertA];
    [alertC addAction:act];
    [self presentViewController:alertC animated:YES completion:nil];
}
//UIImagePickerControllerDelegate
- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<NSString *,id> *)info{
    NSString *mediaType=[info objectForKey:UIImagePickerControllerMediaType];
    
    if ([mediaType isEqualToString:@"public.image"]) {
        UIImage * image;
        // 判断，图片是否允许修改
        if ([picker allowsEditing]){
            //获取用户编辑之后的图像
            image = [info objectForKey:UIImagePickerControllerEditedImage];
        } else {
            // 照片的元数据参数
            image = [info objectForKey:UIImagePickerControllerOriginalImage];
        }
        
        // 压缩图片
        UIImage *compressImg = [self compressPictureWith:image];
        self.imgView.image = compressImg;
        //NSLog(@"%@",NSStringFromCGSize(compressImg.size));
        // 用于上传
//        NSData *tmpData = UIImageJPEGRepresentation(compressImg, 0.5);
        [self.loadimgBtn.imageView setContentMode:UIViewContentModeScaleAspectFit];
        [self.loadimgBtn setBackgroundImage:compressImg forState:UIControlStateNormal];
        [self.loadimgBtn.titleLabel setBackgroundColor:[UIColor whiteColor]];
        [self.loadimgBtn.titleLabel setAlpha:0.7];
        
    }
    [self dismissViewControllerAnimated:YES completion:NULL];
}

- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker{
    [self dismissViewControllerAnimated:YES completion:nil];
}

// 压缩图片
- (UIImage *)compressPictureWith:(UIImage *)originnalImage{
    CGFloat ruleWidth = 2500;
    CGFloat hight;
    if (originnalImage.size.width < ruleWidth) {
        ruleWidth = originnalImage.size.width;
        hight = originnalImage.size.height;
    }else{
        hight = ruleWidth/originnalImage.size.width * originnalImage.size.height;
    }
    
    CGRect rect = CGRectMake(0, 0, ruleWidth, hight);
    // 开启图片上下文
    UIGraphicsBeginImageContext(rect.size);
    // 将图片渲染到图片上下文
    [originnalImage drawInRect:rect];
    // 获取图片
    UIImage *img = UIGraphicsGetImageFromCurrentImageContext();
    // 关闭图片上下文
    UIGraphicsEndImageContext();
    return img;
}

//NSString * docpath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) lastObject]; //Documents目录
//NSString * libpath = [NSSearchPathForDirectoriesInDomains(NSLibraryDirectory, NSUserDomainMask, YES) lastObject]; //Library目录
//NSString * tmp = NSTemporaryDirectory();//tmp目录


- (void)headClick {
    //自定义消息框
    UIAlertController *actionSheet = [UIAlertController alertControllerWithTitle:@"选择" message:@"~" preferredStyle:UIAlertControllerStyleActionSheet];
    actionSheet.popoverPresentationController.sourceView = self.view;
    actionSheet.popoverPresentationController.sourceRect = CGRectMake(self.view.frame.size.width/4,2*(self.view.frame.size.height/3),self.view.frame.size.width/2,1.0);
    
    UIAlertAction *action1 = [UIAlertAction actionWithTitle:@"拍照" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
        NSLog(@"拍照");
        // 判断系统是否支持相机
        UIImagePickerController *imagePickerController = [[UIImagePickerController alloc] init];
        if([UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypeCamera]) {
            imagePickerController.delegate = self; //设置代理
            imagePickerController.allowsEditing = NO;
            imagePickerController.sourceType = UIImagePickerControllerSourceTypeCamera;
            [self presentViewController:imagePickerController animated:YES completion:nil];
        }else{
            [self esayAlert:@"错误" message:@"不支持相机"];
        }
    }];
    UIAlertAction *action2 = [UIAlertAction actionWithTitle:@"从相册选择" style:UIAlertActionStyleDestructive handler:^(UIAlertAction * _Nonnull action) {
        NSLog(@"从相册选择");
        // 判断系统是否支持相机
        UIImagePickerController *imagePickerController = [[UIImagePickerController alloc] init];
        if([UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypePhotoLibrary]) {
            imagePickerController.delegate = self; //设置代理
            imagePickerController.allowsEditing = NO;
            imagePickerController.sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
            [self presentViewController:imagePickerController animated:YES completion:nil];
        }else{
            [self esayAlert:@"错误" message:@"不支持相册"];
        }
    }];
    UIAlertAction *action3 = [UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:^(UIAlertAction * _Nonnull action) {
        NSLog(@"取消");
    }];
    [actionSheet addAction:action1];
    [actionSheet addAction:action2];
    [actionSheet addAction:action3];


    [self presentViewController:actionSheet animated:YES completion:nil];

}


-(void)esayAlert:(NSString*)title message:(NSString*)str {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:title message:str preferredStyle:UIAlertControllerStyleAlert];
    if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPad){
        alert.popoverPresentationController.sourceView = self.view;
        alert.popoverPresentationController.sourceRect = CGRectMake(self.view.frame.size.width/4,2*(self.view.frame.size.height/3),self.view.frame.size.width/2,1.0);
    }
    UIAlertAction *action1 = [UIAlertAction actionWithTitle:@"确定" style:UIAlertActionStyleCancel handler:^(UIAlertAction * _Nonnull action) {}];
    [alert addAction:action1];
    
    [self presentViewController:alert animated:YES completion:nil];
    
}


@end

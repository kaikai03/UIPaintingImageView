//
//  UIView+UIView_dragBar.h
//  PaintView
//
//  Created by kk on 2018/12/15.
//  Copyright © 2018年 kk. All rights reserved.
//

#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@protocol UIDragBarProtocol <NSObject>
@required
- (void)updateThreshold:(NSString*)name threshold:(float)threshold color:(UIColor*)color sender:(id)sender;
- (void)dragBarsuicide;
@end

@interface demo_UIDragBar:UIView
@property (nonatomic, strong) UILabel  * _Nullable text;
@property (nonatomic, strong) NSString  * _Nullable name;

@property (nonatomic, assign) CGFloat threshold;
@property (nonatomic, assign) CGFloat w, hLimited, yBoundary, bottonHeight;
@property (nonatomic, assign) id <UIDragBarProtocol> delegate;
@property (nonatomic, assign) id pFather;


+(UIColor*)calculateCurColor:(float)threshold height:(int)height;
-(BOOL)setNumber:(float)ftext;
-(void)show;
-(void)hide;
@end



NS_ASSUME_NONNULL_END

//
//  UIView+UIView_dragBar.m
//  PaintView
//
//  Created by kk on 2018/12/15.
//  Copyright © 2018年 kk. All rights reserved.
//

#import "demo_UIDragBar.h"

@implementation demo_UIDragBar

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"
- (instancetype)initWithFrame:(CGRect)frame
{
    self = [super initWithFrame:CGRectMake(frame.origin.x, frame.origin.y, frame.size.width, frame.size.height+frame.size.width/2)];
    if (self) {
        self.threshold = 0;
        [self setAlpha:1];
        [self setBackgroundColor:[UIColor clearColor]];
        self.text = [[UILabel alloc]initWithFrame:CGRectMake(frame.size.width/4, 2, frame.size.width/2, frame.size.width/2)];
        [self.text setTextAlignment:NSTextAlignmentCenter];
        [self.text setBackgroundColor:[UIColor clearColor]];
        [self.text setAdjustsFontSizeToFitWidth:YES];
        [self.text setText:@"0.00"];
        [self addSubview:self.text];
        self.layer.cornerRadius = frame.size.width/2;
        self.layer.masksToBounds = YES;
        self.w = self.frame.size.width;
        self.hLimited = frame.size.height;
        self.yBoundary = self.hLimited*0.7;
        self.bottonHeight = self.hLimited*0.3;
        self.hidden = YES;
    }
    return self;
}
#pragma clang diagnostic pop

-(BOOL)setNumber:(float)ftext{
    if (ftext <0 || ftext >5)return NO;
    [self.text setText:[NSString stringWithFormat:@"%.1f",ftext]];
    self.threshold = ftext;
    [self setNeedsDisplay];
    return YES;
}


-(CGPoint)getTouchPoint:(UIEvent *)event{
    
    NSSet *allTouches = [event touchesForView:self];
    UITouch *touch = [allTouches anyObject];
    CGPoint point = [touch locationInView:[touch view]];
    NSLog(@"touch %f, %f", point.x, point.y);
    return point;
}

-(void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event{
    NSLog(@"Began");
}

-(void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event{
    NSLog(@"Moved");
    
    CGPoint point = [self getTouchPoint:event];
    // y [0,h - w/2]
    if (point.y > _yBoundary) {//0~1
        CGFloat tmp = 1 - (point.y - _yBoundary)/_bottonHeight;
        self.threshold = tmp<0?0:tmp;
        [self.text setText:[NSString stringWithFormat:@"%.1f",self.threshold]];
    }else{//1~5
        CGFloat tmp =(1 - point.y/_yBoundary)*4+1;
        self.threshold = tmp>5?5:tmp;
        [self.text setText:[NSString stringWithFormat:@"%.1f",self.threshold]];
    }
    [self setNeedsDisplay];

}

-(void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event{
    if (_delegate != nil && [_delegate respondsToSelector:@selector(dragBarsuicide)]){
        [_delegate dragBarsuicide];
    }
    [self performSelector:@selector(hide) withObject:nil afterDelay:0.2];
}

-(void)touchesCancelled:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event{
    NSLog(@"Cancelled");
}

- (void)removeFromSuperview{
    [self.text removeFromSuperview];
    self.text = nil;
    self.name = nil;
    [super removeFromSuperview];
}

- (void)drawRect:(CGRect)rect{
    NSLog(@"drawRect");
    CGFloat w = self.w;
    CGFloat h = self.hLimited;
    
    CGContextRef context = UIGraphicsGetCurrentContext();

    
    CGContextSetRGBFillColor(context, 0.0, 1.0, 1.0, 1);
    CGContextAddRect(context,  CGRectMake(0, 0, w, h));
    CGContextDrawPath(context, kCGPathEOFill);
    
    CGFloat y;
    UIColor *color;
    if (_threshold < 1) {//0~1
        y = (1 - _threshold)*_bottonHeight+_yBoundary;
        color = [UIColor colorWithRed:0.66*(1-_threshold) green:1.0-0.34*(1-_threshold) blue:0.66*(1-_threshold) alpha:1];
    }else{//1~5
        y = (1 -(_threshold-1)/4)*_yBoundary;
        color = [UIColor colorWithRed:1.0*powf(1-y/_yBoundary,1) green:1.0*powf(y/_yBoundary,3) blue:0.0 alpha:1];
    }
    CGContextSetFillColorWithColor(context,color.CGColor);
    CGContextAddRect(context,  CGRectMake(0, y, w, h-y));
    CGContextDrawPath(context, kCGPathEOFill);
    
    UIGraphicsEndImageContext();
    if (_delegate != nil && [_delegate respondsToSelector:@selector(updateThreshold:threshold:color:sender:)]){
        [_delegate updateThreshold:_name threshold:_threshold color:color sender:_pFather];
    }
}

+(UIColor*)calculateCurColor:(float)threshold height:(int)height{
    float yBoundary = height*0.7;
    UIColor *color;
    if (threshold < 1) {//0~1
        color = [UIColor colorWithRed:0.66*(1-threshold) green:1.0-0.34*(1-threshold) blue:0.66*(1-threshold) alpha:1];
    }else{//1~5
        float y = (1 -(threshold-1)/4)*yBoundary;
        color = [UIColor colorWithRed:1.0*powf(1-y/yBoundary,1) green:1.0*powf(y/yBoundary,3) blue:0.0 alpha:1];
    }
    return color;
}

-(void)show{
    __block float y = self.frame.origin.y;
    __block float height = self.frame.size.height;
    [self setFrame:CGRectMake(self.frame.origin.x,self.frame.origin.y+self.frame.size.height,self.frame.size.width,0)];
    
    [UIView animateWithDuration:.4f animations:^{
        self.hidden = NO;
        [self setFrame:CGRectMake(self.frame.origin.x,y,self.frame.size.width,height)];
     }completion:^(BOOL finished){}];
}

-(void)hide{
    [UIView animateWithDuration:.3f animations:^{
        [self setFrame:CGRectMake(self.frame.origin.x,self.frame.origin.y+self.frame.size.height,self.frame.size.width,0)];
    }completion:^(BOOL finished){
        [self removeFromSuperview];
    }];
}

@end




import CameraModel from '../model/depthanything/CameraModel';
import grantPermission from '../utlis/PermissionUtils';
import Logger from '../utlis/Logger';
import depthanything from 'libdepthanything.so';
import imageNet2012 from '../utlis/ImageNet2012Labels';
import ReturnBtn from '../component/ReturnBtn';
import TopBar from '../component/TopBar';
import display from '@ohos.display';

const TAG: string = '[FSR-AI-CLS]';

@Entry
@Component
struct CameraPage {
  private mXComponentController: XComponentController = new XComponentController();
  private surfaceId: string = '-1';
  private cameraModel: CameraModel = new CameraModel();
  @State modelId: number = 0;
  @State base64: string = "";
  @State fps: number = 0;
  @State swidth: string = "400px"
  @State sheight: string = "240px"
  private w :number =400;
  private h :number =240;
  private scale_factor_x = 0.875;
  private scale_factor_y = 0.875;
  async aboutToAppear() {
    try {
      await grantPermission();
      Logger.info(TAG, `Permission success`);
    } catch (err) {
      Logger.info(TAG, `Permission fail!`);
    }
    let displayClass: display.Display | null = null;
    try {
      displayClass = display.getDefaultDisplaySync();
      this.w = parseInt(displayClass.width.toString()) * 0.7;
      this.h = parseInt(displayClass.height.toString())  ;
      this.swidth = this.w.toString() + "px"
      this.sheight = this.h.toString() + "px"
      this.scale_factor_x = Math.round((this.w / 256) * 100) /100;
      this.scale_factor_y = Math.round((this.h / 256) * 100) /100;
      Logger.info(TAG, displayClass.width.toString() + " " + displayClass.height.toString());
    } catch (exception) {
      console.error('Failed to obtain the default display object. Code: ' + JSON.stringify(exception));
    }
  }

  onPageHide() {
    Logger.info(TAG, 'onPageHide begin');
    Logger.info(TAG, 'onPageHide end');
  }

  onPageShow() {
    let context = getContext()
    let resourceManager = context.resourceManager;
    depthanything.modelDeInit();
    try {
      this.modelId = depthanything.modelInit(resourceManager);
      Logger.info(TAG, `modelId = ${this.modelId.toString()}`)
      if (this.modelId < 0) {
        depthanything.modelDeInit();
      }
    } catch (err) {
      depthanything.modelDeInit();
      Logger.info(TAG, `load model fail`);
    }
    this.surfaceId = this.mXComponentController.getXComponentSurfaceId();
    this.cameraModel.initCamera(this.surfaceId, this.modelId);
    Logger.info(TAG, `aboutToAppear,surfaceId=${this.surfaceId}`);
  }

  build() {
    Column() {
      TopBar({ text: '深度估計', media: $r('app.media.image_classification_icon') })
        .layoutWeight(1)
        .height('30%')
      Row() {
        Column() {
          XComponent({
            id: 'componentId',
            type: 'surface',
            controller: this.mXComponentController
          })
            .onLoad(async () => {
              Logger.info(TAG, 'onLoad is called');
              this.mXComponentController.setXComponentSurfaceSize({ surfaceWidth: 1080, surfaceHeight: 1920 });
              this.surfaceId = this.mXComponentController.getXComponentSurfaceId();
              Logger.info(TAG, `onLoad surfaceId: ${this.surfaceId}`);
              this.cameraModel.initCamera(this.surfaceId, this.modelId);
              this.cameraModel.onImageArrival((base64: string, fps: number) => {
                this.base64 = base64;
                this.fps = fps;
              })
            })

        }
        .height(this.sheight)
        .width(this.swidth)


        Column({ space: 20 }) {
          Text("DepthAnything")
            .fontSize(23)
            .textAlign(TextAlign.Center)
            .backgroundColor('#F1F3F5')
            .width("70%")
            .height("12%")
          Image(this.base64)
            .width("50%")
            .height("40%")
          ReturnBtn()
        }
        .align(Alignment.Top)
        .width("30%")
      }
      .height("70%")
    }
    .backgroundColor('#00fcfafa')
    .width('100%')
    .height('100%')
  }

  async aboutToDisappear() {
    await this.cameraModel.cameraRelease();
  }
}
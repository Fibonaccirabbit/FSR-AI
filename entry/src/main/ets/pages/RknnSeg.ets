

import CameraModel from '../model/rknnseg/CameraModel';
import Logger from '../utlis/Logger';
import rknnSocket from '../model/rknnsocket/rknnSokcet';
import socket from '@ohos.net.socket';
import TopBar from '../component/TopBar';
import ReturnBtn from '../component/ReturnBtn';
import display from '@ohos.display';

const TAG: string = '[FSR-AI-PPSEG]';

@Entry
@Component
struct CameraPage {
  private mXComponentController: XComponentController = new XComponentController();
  private surfaceId: string = '-1';
  private cameraModel: CameraModel = new CameraModel();
  @State modelId: number = 0;
  @State fps: number = 0;
  @State base64: string = "";
  public socket: rknnSocket = new rknnSocket("127.0.0.1", 12600);
  @State swidth: string = "400px"
  @State sheight: string = "240px"
  private w :number =400;
  private h :number =240;
  private scale_factor_x = 0.875;
  private scale_factor_y = 0.875;
  aboutToAppear() {
    this.onRknnMsg();
    let displayClass: display.Display | null = null;
    try {
      displayClass = display.getDefaultDisplaySync();
      this.w = parseInt(displayClass.width.toString()) * 0.7;
      this.h = parseInt(displayClass.height.toString())  ;
      this.swidth = this.w.toString() + "px"
      this.sheight = this.h.toString() + "px"
      this.scale_factor_x = Math.round((this.w / 640) * 100) /100;
      this.scale_factor_y = this.scale_factor_x;
      Logger.info(TAG, displayClass.width.toString() + " " + displayClass.height.toString());
    } catch (exception) {
      console.error('Failed to obtain the default display object. Code: ' + JSON.stringify(exception));
    }
  }

  onRknnMsg() {
    this.socket.rknnServerPort = 12800;
    this.socket.bindLocal();
    this.socket.tcpServer.on('message', (value: SocketInfo) => {
      Logger.info(TAG, "on message");
      let buffer = value.message;
      let dataView = new DataView(buffer);
      let str = "";
      for (let i = 0; i < dataView.byteLength; ++i) {
        str += String.fromCharCode(dataView.getUint8(i));
      }
      this.base64 = str;
      Logger.info(TAG, "on connect received:" + str);
    });
  }

  build() {
    Column() {
      TopBar({ text: '目标分割', media: $r('app.media.obj_detection_icon') })
        .layoutWeight(1)
        .height('30%')
      Row() {
        Stack() {
          Image(this.base64)
            .zIndex(2)
            .opacity(0.5)
            .objectFit(ImageFit.Cover)
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
              this.cameraModel.initCamera(this.surfaceId, this.modelId, this.socket);
              this.cameraModel.onImageArrival((value: number, fps: number) => {
              })
            })
        }
        .backgroundColor("#ffc63333")
        .width(this.swidth)
        .height(this.sheight)

        Column({ space: 20 }) {
          Text("PPSEG")
            .fontSize(42)
            .textAlign(TextAlign.Center)
            .backgroundColor('#F1F3F5')
            .width("70%")
            .height("22%")
          ReturnBtn()
        }
        .width("30%")
        .align(Alignment.Top)

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

class SocketInfo {
  message: ArrayBuffer = new ArrayBuffer(1);
  remoteInfo: socket.SocketRemoteInfo = {} as socket.SocketRemoteInfo;
}
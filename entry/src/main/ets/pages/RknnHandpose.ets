

import CameraModel from '../model/rknnhandpose/CameraModel';
import Logger from '../utlis/Logger';
import util from '@ohos.util';
import rknnSocket from '../model/rknnsocket/rknnSokcet';
import socket from '@ohos.net.socket';
import TopBar from '../component/TopBar';
import ReturnBtn from '../component/ReturnBtn';
import display from '@ohos.display';

const TAG: string = '[FSR-AI-RKNN]';

interface point {
  x?: number,
  y?: number
}

@Entry
@Component
struct CameraPage {
  private mXComponentController: XComponentController = new XComponentController();
  private surfaceId: string = '-1';
  private cameraModel: CameraModel = new CameraModel();
  @State xmin: number = 0;
  @State ymin: number = 0;
  @State xmax: number = 0;
  @State ymax: number = 0;
  @State pointData: point[] = [{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0},{x:0,y:0}];
  @State colorSet: Color[] = [Color.Red, Color.Green, Color.Yellow, Color.Blue];
  @State indexSet: number[] = [1, 2, 3];
  @State modelId: number = 0;
  @State fps: number = 0;
  @State swidth: string = "400px"
  @State sheight: string = "240px"
  private w :number =400;
  private h :number =240;
  private scale_factor_x = 0.875;
  private scale_factor_y = 0.875;
  @State mouse_x : number = 0;
  @State mouse_y : number  = 0;
  private pre_local = {x:0,y:0};
  public socket: rknnSocket = new rknnSocket("127.0.0.1", 12600);

  aboutToAppear() {
    this.onRknnMsg();
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

  onRknnMsg() {
    this.socket.rknnServerPort = 12900;
    this.socket.bindLocal();
    this.socket.tcpServer.on('message', (value: SocketInfo) => {
      Logger.info(TAG, "on message");
      let buffer = value.message;
      let dataView = new DataView(buffer);
      let str = "";
      for (let i = 0; i < dataView.byteLength; ++i) {
        str += String.fromCharCode(dataView.getUint8(i));
      }
      this.resolveRknnHandposeResult(str);
      Logger.info(TAG, "on connect received:" + str);
    });
  }

  resolveRknnHandposeResult(result: string) {
    Logger.info(TAG, result)
    let targets: string[] = result.split("&");

    let rknnData: point [] = [];
    this.pre_local.x = this.pointData[8].x;
    this.pre_local.y = this.pointData[8].y;
    for (let i = 0;i < 21; i++) {
      let p: point  = {} as point ;
      p.x = parseFloat(targets[i * 2 + 0]);
      p.y = parseFloat(targets[i * 2 + 1]);
      Logger.info(TAG, JSON.stringify(p));
      rknnData.push(p);
    }
    Logger.info(TAG, JSON.stringify(rknnData));
    this.pointData = rknnData;
    this.mouse_x = this.mouse_x + (this.pointData[8].x - this.pre_local.x) * 8;
    this.mouse_y = this.mouse_y + (this.pointData[8].y - this.pre_local.y) * 6;
  }

  build() {
    Column(){
      TopBar({ text: '关键点检测', media: $r('app.media.hand_pose') })
        .layoutWeight(1)
        .height('30%')
      Stack() {
        Row() {
          Stack() {
            ForEach(this.pointData, (item: point, index) => {
              Column() {
                Blank()
                  .backgroundColor(this.colorSet[index % 4])
                  .width(10)
                  .height(10)
                  .borderRadius(5)
              }
              .position({ x: px2vp(this.w - item.x*this.scale_factor_x), y: px2vp(item.y*this.scale_factor_y) })
              .zIndex(index + 1)
            })
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
            Text("handpose")
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
      }
      .height("70%")
    }
    .backgroundColor('#00fcfafa')
    .width('100%')
    .height('100%')
  }

  uint8ArrayToString(uint8Array: Uint8Array): string {
    let textDecoder = util.TextDecoder.create('utf-8');
    return textDecoder.decodeWithStream(uint8Array);
  }

  uint8ArrayToStringTest(u8a: Uint8Array) {
    let dataStr = "";
    for (let i = 0; i < u8a.length; i++) {
      dataStr += String.fromCharCode(u8a[i])
    }
    return dataStr;
  }

  async aboutToDisappear() {
    await this.cameraModel.cameraRelease();
  }
}

class SocketInfo {
  message: ArrayBuffer = new ArrayBuffer(1);
  remoteInfo: socket.SocketRemoteInfo = {} as socket.SocketRemoteInfo;
}


import CameraModel from '../model/rknnface/CameraModel';
import Logger from '../utlis/Logger';
import util from '@ohos.util';
import emitter from "@ohos.events.emitter";
import display from '@ohos.display';
import rknnSocket from '../model/rknnsocket/rknnSokcet';
import socket from '@ohos.net.socket';
import ReturnBtn from '../component/ReturnBtn';
import TopBar from '../component/TopBar';

const TAG: string = '[FSR-AI-RetinaFace]';

interface ObjectDesc {
  xmin?: number;
  ymin?: number;
  xmax?: number;
  ymax?: number;
  label?: string;
  prop?: number;
  px1?: number;
  py1?: number;
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
  @State pointData: ObjectDesc[] = [];
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
  public socket: rknnSocket = new rknnSocket("127.0.0.1", 12600);

  aboutToAppear() {
    // 收到eventId为1的事件后执行该回调
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
    this.socket.rknnServerPort = 13000;
    this.socket.bindLocal();
    this.socket.tcpServer.on('message', (value: SocketInfo) => {
      Logger.info(TAG, "on message");
      let buffer = value.message;
      let dataView = new DataView(buffer);
      let str = "";
      for (let i = 0; i < dataView.byteLength; ++i) {
        str += String.fromCharCode(dataView.getUint8(i));
      }
      // this.base64 = str;
      this.resolveRknnFaceResult(str);
      Logger.info(TAG, "on connect received:" + str);
    });
  }

  resolveRknnFaceResult(result: string) {
    let targets: string[] = result.split("@");
    let rknnData: ObjectDesc [] = [];
    targets.forEach((target: string) => {
      const [label, xmin, ymin, xmax, ymax, prop, px1, py1] = target.split('&');
      if (label != "") {
        let t: ObjectDesc = {} as ObjectDesc;
        t.xmin = parseInt(xmin);
        t.ymin = parseInt(ymin);
        t.xmax = parseInt(xmax);
        t.ymax = parseInt(ymax);
        t.label = label;
        t.prop = parseFloat(prop) * 100
        t.px1 = parseInt(px1);
        t.py1 = parseInt(py1);
        rknnData.push(t);
      }
    })
    this.pointData = rknnData;
  }

  build() {
    Column() {
      TopBar({ text: '人脸识别', media: $r('app.media.face_icon') })
        .layoutWeight(1)
        .height('30%')
      Row() {
        Stack() {
          ForEach(this.pointData, (item: ObjectDesc, index) => {
            Column() {
              Text(item.label)
                .fontSize(24)
                .fontColor(this.colorSet[index % 4])
              Text(String(item.prop).substring(0, 4) + '%')
                .fontSize(12)
                .fontColor(this.colorSet[index % 4])
            }
            .position({ x: px2vp(this.w - item.xmax * this.scale_factor_x), y: px2vp(item.ymin * this.scale_factor_y) })
            .width(px2vp((item.xmax - item.xmin) * this.scale_factor_x))
            .height(px2vp((item.ymax - item.ymin) * this.scale_factor_y))
            .borderWidth(2)
            .borderColor(this.colorSet[index % 4])
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
          Text("RetinaFace")
            .fontSize(32)
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
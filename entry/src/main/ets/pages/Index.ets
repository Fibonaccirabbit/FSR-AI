import hilog from '@ohos.hilog';
import mobilenetV2 from 'libmobilenetV2.so'
import ssd300 from "libssd300.so"
import fpr from "libfpr.so"
import abilityAccessCtrl from '@ohos.abilityAccessCtrl';
import Logger from '../utlis/Logger';
import image from '@ohos.multimedia.image';
import router from '@ohos.router';
import rknnSocket from '../model/rknnsocket/rknnSokcet';
import  { flowerBase64, someBase64 } from '../utlis/Base64Image';

let TAG = "[FSR-AI]"

@Entry
@Component
struct Index {
  @State message: string = 'Hello World'
  @State pixel: PixelMap = undefined;
  private socket: rknnSocket = new rknnSocket("127.0.0.1", 12600);

  async aboutToAppear() {
    let AtManager = abilityAccessCtrl.createAtManager();
    AtManager.requestPermissionsFromUser(getContext(this), ['ohos.permission.READ_MEDIA', 'ohos.permission.MEDIA_LOCATION', "ohos.permission.LOCATION", "ohos.permission.CAMERA", "ohos.permission.INTERNET"])
  }

  build() {
    Row() {
      Column() {
        Text("视频流RKNN-PPSeg")
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(() => {
            router.pushUrl({
              url: 'pages/RknnSeg' // 目标url
            })
          })
        Text("视频流RKNN-Yolov5")
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(() => {
            router.pushUrl({
              url: 'pages/RknnYolo' // 目标url
            })
          })
        Text("视频流mobilenetV2")
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(() => {
            router.pushUrl({
              url: 'pages/Classification' // 目标url
            })
          })
        Text("视频流ssd300")
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(() => {
            router.pushUrl({
              url: 'pages/ObjectDetect' // 目标url
            })
          })
        Text("启动模型")
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(() => {
            let res: number = fpr.modelInit(getContext(this).resourceManager);
            Logger.info(TAG, res.toString())
          })
        Text("推理模型")
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(() => {
            this.inferenceHSI()
          })
        Text("关闭模型")
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(() => {
            let res: number = fpr.modelDeInit();
            Logger.info(TAG, res.toString())
          })

        Text("socket测试")
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(() => {
            this.socket.prepare();
          })
        Text("socket发送")
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(() => {
            this.sendImage();
          })
        Text("socket测试解析")
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(() => {
            this.socket.resolveRknnYoloResult("bird&129&49&512&460@");
          })
        Image(someBase64)
      }
      .width('100%')
    }
    .height('100%')
  }

  async generatePic() {
    let globalContext = getContext(this)
    let inputBuffer: Uint8Array = new Uint8Array();
    let inputName: string = 'flower.jpg';
    let imageSourceApi: image.ImageSource = image.createImageSource(new ArrayBuffer(96));

    await globalContext.resourceManager.getRawFileContent(inputName).then((buffer: Uint8Array) => {
      inputBuffer = buffer;
      let sourceOptions: image.SourceOptions = {
        sourceDensity: 120,
        sourcePixelFormat: 3,
        sourceSize: { height: 480, width: 640 }
      };
      imageSourceApi = image.createImageSource(buffer.buffer, sourceOptions);
      console.log(TAG + '=========input image byte length: ' + buffer.byteLength)
    })
    let decodingOptions: image.DecodingOptions = {
      sampleSize: 1,
      editable: true,
      desiredSize: { height: 480, width: 640 },
      desiredPixelFormat: 3,
    };
    let pixel: PixelMap = await imageSourceApi.createPixelMap(decodingOptions)
    // 场景一：将读取的整张图像像素数据结果写入ArrayBuffer中
    let readBuffer: ArrayBuffer = new ArrayBuffer(pixel.getPixelBytesNumber());
    await pixel.readPixelsToBuffer(readBuffer).then(() => {
      console.log(TAG + '=========无损 input image byte length: ' + readBuffer.byteLength)
      console.info(TAG + 'Succeeded in reading image pixel data.');
    }).catch((error: string) => {
      console.error(TAG + 'Failed to read image pixel data. And the error is: ' + JSON.stringify(error));
    })

    return readBuffer
  }

  async inference() {
    let inputData = await this.generatePic();
    console.info(TAG + "输入数据长度" + inputData.byteLength)
    let f32 = new Float32Array(500 * 374 * 4);
    let temp = new Uint8Array(inputData)
    // for (let i = 0;i < 100; i++) {
    //   console.info(this.TAG+temp[i])
    // }
    f32.set(new Float32Array(new Uint8Array(inputData)))
    console.info(TAG + f32[f32.length-1])
    // for(let i =0;i<10;i++){
    //   console.info(this.TAG+uint8[i])
    // }
    hilog.info(0x0000, TAG, '模型推理结果: %{public}d', mobilenetV2.modelInference(f32.buffer));
  }

  async inferenceHSI() {
    let inputData = await this.generatePic();
    console.info(TAG + "输入数据长度" + inputData.byteLength)
    let f32 = new Float32Array(500 * 374 * 4);
    let temp = new Uint8Array(inputData)
    // for (let i = 0;i < 100; i++) {
    //   console.info(this.TAG+temp[i])
    // }
    f32.set(new Float32Array(new Uint8Array(inputData)))
    console.info(TAG + f32[f32.length-1])
    // for(let i =0;i<10;i++){
    //   console.info(this.TAG+uint8[i])
    // }
    let buffer: ArrayBuffer = fpr.modelInference(f32.buffer);
    Logger.info(TAG, "exit")
    let opts: image.InitializationOptions = { editable: true, pixelFormat: 3, size: { height: 224, width: 224 } }
    image.createPixelMap(buffer, opts).then((pixelMap: image.PixelMap) => {

      pixelMap.writeBufferToPixels(buffer).then(() => {
        this.pixel = pixelMap;
        console.info("Succeeded in writing data from a buffer to a PixelMap.");
      }).catch((error: string) => {
        console.error("Failed to write data from a buffer to a PixelMap.");
      })
      console.info('Succeeded in creating pixelmap.');
    }).catch((error: string) => {
      console.error('Failed to create pixelmap.');
    })
    // hilog.info(0x0000, TAG, '模型推理结果: %{public}d', fpr.modelInference(f32.buffer));
  }

  async sendImage() {
    let inputData = await this.generatePic();
    console.info(TAG + "输入数据长度" + inputData.byteLength)
    this.socket.sendImage(inputData);
  }
}

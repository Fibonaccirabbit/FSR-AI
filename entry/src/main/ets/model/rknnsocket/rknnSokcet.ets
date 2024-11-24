import socket from '@ohos.net.socket';
import Logger from '../../utlis/Logger';
import { BusinessError } from '@ohos.base';
import emitter from '@ohos.events.emitter';

interface ObjectDesc {
  xmin?: number;
  ymin?: number;
  xmax?: number;
  ymax?: number;
  label?: string;
  prop?: number;
}
// 定义一个eventId为1的事件
let RknnYoloEvent: emitter.InnerEvent = {
  eventId: 3
};

let RknnSegEvent: emitter.InnerEvent = {
  eventId: 4
};

let TAG = "[FSR-SOCKET]"

export default class rknnSocket {
  public tcpServer: socket.TCPSocket;
  private ipAddress: socket.NetAddress;
  private address = "192.168.50.59";
  private port = 12700;
  public rknnServerIP = "127.0.0.1"
  public rknnServerPort = 13000;
  private preTime = 0;
  private curTime = 0;
  private fps;

  constructor(ip: string, port: number) {
    this.tcpServer = socket.constructTCPSocketInstance();
    this.ipAddress = {} as socket.NetAddress;
    this.ipAddress.address = ip;
    this.ipAddress.port = port;
  }

  prepare() {
    // 创建一个TCPSocket连接，返回一个TCPSocket对象。
    this.tcpServer = socket.constructTCPSocketInstance();
    this.tcpServer.bind(this.ipAddress, (err: BusinessError) => {
      if (err) {

        Logger.info(TAG, "bind fail");
        return;
      }
      Logger.info(TAG, "bind success");
      this.connectRknnServer(this.rknnServerIP, this.rknnServerPort);
    });

    this.tcpServer.on('message', (value: SocketInfo) => {
      Logger.info(TAG, "on message");
      let buffer = value.message;
      let dataView = new DataView(buffer);
      let str = "";
      for (let i = 0; i < dataView.byteLength; ++i) {
        str += String.fromCharCode(dataView.getUint8(i));
      }
      Logger.info(TAG, "on connect received:" + str);
      this.curTime = new Date().getTime();
      this.fps = Math.floor(1000 / (this.curTime - this.preTime));
      this.preTime = this.curTime;
      this.resolveRknnYoloResult(str);
    });
    this.tcpServer.on('connect', () => {
      Logger.info(TAG, "on connect");

    });
    this.tcpServer.on('close', () => {
      Logger.info(TAG, "on close");
    });
  }

  bindLocal() {
    this.tcpServer = socket.constructTCPSocketInstance();
    this.tcpServer.bind(this.ipAddress, (err: BusinessError) => {
      if (err) {
        Logger.info(TAG, "bind fail");
        return;
      }
      Logger.info(TAG, "bind success");
      this.connectRknnServer(this.rknnServerIP, this.rknnServerPort);
    });
  }

  connectRknnServer(remoteIP: string, port: number) {
    // 连接到指定的IP地址和端口。
    let ipAddress: socket.NetAddress = {} as socket.NetAddress;
    ;
    ipAddress.address = remoteIP;
    ipAddress.port = port;

    let tcpConnect: socket.TCPConnectOptions = {} as socket.TCPConnectOptions;
    tcpConnect.address = ipAddress;
    tcpConnect.timeout = 6000;

    this.tcpServer.connect(tcpConnect, (err: BusinessError) => {
      if (err) {
        Logger.info(TAG, "connect fail");
        return;
      }
      Logger.info(TAG, "connect success");
    });
  }

  sendBase64Image() {
    // 发送数据
    let tcpSendOptions: socket.TCPSendOptions = {} as socket.TCPSendOptions;
    tcpSendOptions.data = 'Hello, server!';
    this.tcpServer.send(tcpSendOptions, (err: BusinessError) => {
      if (err) {
        console.log('send fail');
        return;
      }
      console.log('send success');
    })
  }

  sendImage(buffer: ArrayBuffer) {
    // 发送数据
    let tcpSendOptions: socket.TCPSendOptions = {} as socket.TCPSendOptions;
    tcpSendOptions.data = buffer;
    this.tcpServer.send(tcpSendOptions, (err: BusinessError) => {
      if (err) {
        Logger.info(TAG, "send fail");
        this.connectRknnServer(this.rknnServerIP, this.rknnServerPort)
        return;
      }
      Logger.info(TAG, "send success");

    })
  }

  resolveRknnYoloResult(result: string) {
    let targets: string[] = result.split("@");
    let rknnData: ObjectDesc [] = [];
    targets.forEach((target: string) => {
      const [label, xmin, ymin, xmax, ymax, prop] = target.split('&');
      if (label != "") {
        let t: ObjectDesc = {} as ObjectDesc;
        t.xmin = parseInt(xmin);
        t.ymin = parseInt(ymin);
        t.xmax = parseInt(xmax);
        t.ymax = parseInt(ymax);
        t.label = label;
        t.prop = parseFloat(prop) * 100

        rknnData.push(t);
      }
    })
    Logger.info(TAG, JSON.stringify(rknnData));
    let eventData = {
      data: {
        "content": JSON.stringify(rknnData),
        "fps": JSON.stringify(this.fps),
        "id": 1,
        "isEmpty": false,
      }
    };
    // 发送eventId为1的事件，事件内容为eventData
    emitter.emit(RknnYoloEvent, eventData);
  }

  resolveRknnSegResult(result: string) {

    Logger.info(TAG, JSON.stringify(result));
    let eventData = {
      data: {
        "content": result.split(",")[1],
        "fps": JSON.stringify(this.fps),
        "id": 1,
        "isEmpty": false,
      }
    };
    // 发送eventId为1的事件，事件内容为eventData
    emitter.emit(RknnSegEvent, eventData);
  }

  close() {
    this.tcpServer.close();
  }
}

class SocketInfo {
  message: ArrayBuffer = new ArrayBuffer(1);
  remoteInfo: socket.SocketRemoteInfo = {} as socket.SocketRemoteInfo;
}
import UIAbility from '@ohos.app.ability.UIAbility';
import hilog from '@ohos.hilog';
import window from '@ohos.window';
import { BusinessError } from '@ohos.base';

export default class EntryAbility extends UIAbility {
  onCreate(want, launchParam) {
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onCreate');
  }

  onDestroy() {
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onDestroy');
  }

  onWindowStageCreate(windowStage: window.WindowStage) {
    // Main window is created, set main page for this ability
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onWindowStageCreate');
    let windowClass: window.Window | null = null;
    windowStage.getMainWindow((err: BusinessError, data) => {
      let errCode: number = err.code;
      if (errCode) {
        console.error('Failed to obtain the main window. Cause: ' + JSON.stringify(err));
        return;
      }
      windowClass = data;
      console.info('Succeeded in obtaining the main window. Data: ' + JSON.stringify(data));

      // 2.实现沉浸式效果。方式一：设置导航栏、状态栏不显示。
      let names: Array<'status' | 'navigation'> = [];
      windowClass.setWindowSystemBarEnable(names, (err: BusinessError) => {
        let errCode: number = err.code;
        if (errCode) {
          console.error('Failed to set the system bar to be visible. Cause:' + JSON.stringify(err));
          return;
        }
        console.info('Succeeded in setting the system bar to be visible.');
      });
      // // 2.实现沉浸式效果。方式二：设置窗口为全屏布局，配合设置导航栏、状态栏的透明度、背景/文字颜色及高亮图标等属性，与主窗口显示保持协调一致。
      // let isLayoutFullScreen = true;
      // windowClass.setWindowLayoutFullScreen(isLayoutFullScreen, (err: BusinessError) => {
      //   let errCode: number = err.code;
      //   if (errCode) {
      //     console.error('Failed to set the window layout to full-screen mode. Cause:' + JSON.stringify(err));
      //     return;
      //   }
      //   console.info('Succeeded in setting the window layout to full-screen mode.');
      // });
      // let sysBarProps: window.SystemBarProperties = {
      //   statusBarColor: '#ff00ff',
      //   navigationBarColor: '#00ff00',
      //   // 以下两个属性从API Version 8开始支持
      //   statusBarContentColor: '#ffffff',
      //   navigationBarContentColor: '#ffffff'
      // };
      // windowClass.setWindowSystemBarProperties(sysBarProps, (err: BusinessError) => {
      //   let errCode: number = err.code;
      //   if (errCode) {
      //     console.error('Failed to set the system bar properties. Cause: ' + JSON.stringify(err));
      //     return;
      //   }
      //   console.info('Succeeded in setting the system bar properties.');
      // });
    })
    windowStage.loadContent('pages/Home', (err, data) => {
      if (err.code) {
        hilog.error(0x0000, 'testTag', 'Failed to load the content. Cause: %{public}s', JSON.stringify(err) ?? '');
        return;
      }
      hilog.info(0x0000, 'testTag', 'Succeeded in loading the content. Data: %{public}s', JSON.stringify(data) ?? '');
    });
  }

  onWindowStageDestroy() {
    // Main window is destroyed, release UI related resources
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onWindowStageDestroy');
  }

  onForeground() {
    // Ability has brought to foreground
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onForeground');
  }

  onBackground() {
    // Ability has back to background
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onBackground');
  }
};

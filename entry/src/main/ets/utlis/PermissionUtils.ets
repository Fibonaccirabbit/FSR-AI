/*
 * Copyright 2023 Unionman Technology Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import abilityAccessCtrl from '@ohos.abilityAccessCtrl';
import type { Permissions } from '@ohos.abilityAccessCtrl';
import bundleManager from '@ohos.bundle.bundleManager';
import Logger from './Logger';
import { BusinessError } from '@ohos.base';

const TAG: string = '[Permission]';

const PERMISSIONS: Array<Permissions> = [
  'ohos.permission.CAMERA',
  'ohos.permission.MEDIA_LOCATION',
  'ohos.permission.MICROPHONE',
  'ohos.permission.READ_MEDIA',
  'ohos.permission.WRITE_MEDIA'
];

export default async function grantPermission(): Promise<boolean> {
  try {
    let atManager = abilityAccessCtrl.createAtManager();
    let pems: Array<Permissions> = [];

    for (let i = 0; i < PERMISSIONS.length; i++) {
      let state = await checkAccessToken(PERMISSIONS[i]);
      Logger.info(TAG, `grantPermission  checkAccessToken ${PERMISSIONS[i]} + : ${JSON.stringify(state)}`);
      if (state !== abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED) {
        pems.push(PERMISSIONS[i]);
      }
    }
    Logger.info(TAG, 'pems.length:' + JSON.stringify(pems.length));
    if (pems.length > 0) {
      Logger.info(TAG, 'grantPermission  requestPermissionsFromUser :' + JSON.stringify(pems));
      let result = await atManager.requestPermissionsFromUser(globalThis.cameraContext, pems);
      Logger.info(TAG, `contextTest ${JSON.stringify(globalThis.cameraContext)}`)
      let grantStatus: Array<number> = result.authResults;
      let length: number = grantStatus.length;
      for (let i = 0; i < length; i++) {
        Logger.info(TAG, `grantPermission  requestPermissionsFromUser ${result.permissions[i]} + : ${grantStatus[i]}`);

        if (grantStatus[i] === 0) {
          // 用户授权，可以继续访问目标操作
        } else {
          // 用户拒绝授权，提示用户必须授权才能访问当前页面的功能
          console.log(TAG + 'grantPermission  fail ');
          return false;
        }
      }
    }
    // 授权成功
    Logger.info(TAG, 'grantPermission  success ');
    return true;
  } catch (e) {
    Logger.info(TAG, 'grantPermission  fail ');
    return false;
  }
}

async function checkPermissions(): Promise<void> {
  const permissions: Array<Permissions> = ['ohos.permission.READ_CALENDAR'];
  let grantStatus: abilityAccessCtrl.GrantStatus = await checkAccessToken(permissions[0]);

  if (grantStatus === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED) {
    // 已经授权，可以继续访问目标操作
  } else {
    // 申请日历权限
  }
}


async function checkAccessToken(permission: Permissions): Promise<abilityAccessCtrl.GrantStatus> {
  let atManager: abilityAccessCtrl.AtManager = abilityAccessCtrl.createAtManager();
  let grantStatus: abilityAccessCtrl.GrantStatus = abilityAccessCtrl.GrantStatus.PERMISSION_DENIED;

  // 获取应用程序的accessTokenID
  let tokenId: number = 0;
  try {
    let bundleInfo: bundleManager.BundleInfo = await bundleManager.getBundleInfoForSelf(bundleManager.BundleFlag.GET_BUNDLE_INFO_WITH_APPLICATION);
    let appInfo: bundleManager.ApplicationInfo = bundleInfo.appInfo;
    tokenId = appInfo.accessTokenId;
  } catch (error) {
    let err: BusinessError = error as BusinessError;
    console.error(`Failed to get bundle info for self. Code is ${err.code}, message is ${err.message}`);
  }

  // 校验应用是否被授予权限
  try {
    grantStatus = atManager.checkAccessTokenSync(tokenId, permission);
    Logger.info(TAG, `${permission} grantStatus = ${grantStatus}`)
  } catch (error) {
    let err: BusinessError = error as BusinessError;
    console.error(`Failed to check access token. Code is ${err.code}, message is ${err.message}`);
  }

  return grantStatus;
}

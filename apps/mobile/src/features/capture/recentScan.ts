import AsyncStorage from "@react-native-async-storage/async-storage";

const RECENT_SCAN_KEY = "video-to-minecraft.recent-scan-id";

export async function saveRecentScanId(scanId: string): Promise<void> {
  await AsyncStorage.setItem(RECENT_SCAN_KEY, scanId);
}

export async function getRecentScanId(): Promise<string | null> {
  return AsyncStorage.getItem(RECENT_SCAN_KEY);
}

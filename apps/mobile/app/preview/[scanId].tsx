import { Redirect, useLocalSearchParams } from "expo-router";

import { PreviewScreen } from "@/features/preview/PreviewScreen";

export default function PreviewRoute() {
  const { scanId } = useLocalSearchParams<{ scanId?: string | string[] }>();
  if (!scanId || Array.isArray(scanId)) return <Redirect href="/" />;
  return <PreviewScreen scanId={scanId} />;
}

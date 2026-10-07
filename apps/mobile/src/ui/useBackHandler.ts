import { useEffect, useRef } from "react";
import { BackHandler } from "react-native";

/**
 * Routes the Android back button/gesture to `onBack` while mounted. Screens kept in
 * local state (not routes) need this, or back closes the app. Return false from
 * `onBack` to fall through to the default behaviour; omit it to block back.
 */
export function useBackHandler(onBack?: () => boolean | void) {
  const latest = useRef(onBack);
  latest.current = onBack;
  useEffect(() => {
    const subscription = BackHandler.addEventListener("hardwareBackPress", () => {
      const handler = latest.current;
      return handler ? handler() !== false : true;
    });
    return () => subscription.remove();
  }, []);
}

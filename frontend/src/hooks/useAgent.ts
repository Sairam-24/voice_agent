import { useEffect, useState, useSyncExternalStore } from "react";
import { VoiceController } from "../controller.ts";

/** Creates the controller once, starts and stops it with the component, and subscribes to its store. */
export function useAgent() {
  const [controller] = useState(() => new VoiceController());
  useEffect(() => {
    controller.start();
    return () => controller.dispose();
  }, [controller]);
  const state = useSyncExternalStore(controller.store.subscribe, controller.store.getState);
  return { controller, state };
}

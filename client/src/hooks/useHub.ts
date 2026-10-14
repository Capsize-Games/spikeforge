import { useCallback, useEffect } from "react";

import type { HubQueryInput } from "../hubTypes";
import { useWebSocket } from "../useWebSocket";
import { useHubState } from "./useHubState";

type WebSocketControls = ReturnType<typeof useWebSocket>;

interface Options {
  ws: WebSocketControls;
  connected: boolean;
}

/** Hub catalog actions over the shared hub browser state. */
export function useHub({ ws, connected }: Options) {
  const { sendHub } = ws;
  const hub = useHubState();
  const {
    setSearch,
    setDownload,
    setInspect,
    setImported,
    setPull,
    setSelected,
  } = hub;

  /** Refresh the catalog, optionally re-filtering it. */
  const refresh = useCallback(
    (filters: HubQueryInput = {}) => {
      setSearch(null);
      sendHub("hub_list", filters);
    },
    [sendHub, setSearch],
  );

  const runSearch = useCallback(
    (query: string, limit = 20) => sendHub("hub_search", { query, limit }),
    [sendHub],
  );

  const select = useCallback(
    (id: string) => {
      setSelected(id);
      setInspect(null);
      setImported(null);
    },
    [setImported, setInspect, setSelected],
  );

  const downloadEntry = useCallback(
    (id: string) => {
      select(id);
      setDownload({
        id,
        status: "downloading",
        bytes: 0,
        total_bytes: null,
        verified: false,
      });
      sendHub("hub_download", {}, id);
    },
    [select, sendHub, setDownload],
  );

  const cancelDownload = useCallback(
    () => sendHub("hub_cancel", {}),
    [sendHub],
  );

  const inspectEntry = useCallback(
    (id: string) => {
      setSelected(id);
      setImported(null);
      sendHub("hub_inspect", {}, id);
    },
    [sendHub, setImported, setSelected],
  );

  const importEntry = useCallback(
    (id: string) => {
      setSelected(id);
      sendHub("hub_import", {}, id);
    },
    [sendHub, setSelected],
  );

  const pullEntry = useCallback(
    (id: string) => {
      select(id);
      setPull({
        id,
        status: "downloading",
        bytes: 0,
        total_bytes: null,
        verified: false,
      });
      sendHub("hub_pull", {}, id);
    },
    [select, sendHub, setPull],
  );

  const buildBundle = useCallback(
    (name: string) => sendHub("bundle_build", {}, name),
    [sendHub],
  );

  useEffect(() => {
    if (!connected) return;
    sendHub("hub_list", {});
  }, [connected, sendHub]);

  return {
    list: hub.list,
    search: hub.search,
    download: hub.download,
    inspect: hub.inspect,
    imported: hub.imported,
    pull: hub.pull,
    built: hub.built,
    selected: hub.selected,
    onMessage: hub.onMessage,
    refresh,
    runSearch,
    select,
    downloadEntry,
    cancelDownload,
    inspectEntry,
    importEntry,
    pullEntry,
    buildBundle,
  };
}

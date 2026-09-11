import { useCallback, useEffect, useState } from "react";

import type {
  HubDownloadState,
  HubImportPayload,
  HubInspectPayload,
  HubListPayload,
  HubQueryInput,
  HubSearchPayload,
} from "../hubTypes";
import type { ServerMsg } from "../types";
import { useWebSocket } from "../useWebSocket";

type WebSocketControls = ReturnType<typeof useWebSocket>;

interface Options {
  ws: WebSocketControls;
  connected: boolean;
}

/** Own the hub browser state and fold hub server messages into it. */
export function useHub({ ws, connected }: Options) {
  const { sendHub } = ws;
  const [list, setList] = useState<HubListPayload | null>(null);
  const [search, setSearch] = useState<HubSearchPayload | null>(null);
  const [download, setDownload] = useState<HubDownloadState | null>(null);
  const [inspect, setInspect] = useState<HubInspectPayload | null>(null);
  const [imported, setImported] = useState<HubImportPayload | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  /** Fold one server message into the hub state; other types pass through. */
  const onMessage = useCallback((msg: ServerMsg) => {
    switch (msg.type) {
      case "hub_list":
        setList(msg.payload);
        break;
      case "hub_search":
        setSearch(msg.payload);
        break;
      case "hub_download_state":
        setDownload(msg.payload);
        break;
      case "hub_inspect":
        setInspect(msg.payload);
        break;
      case "hub_import":
        setImported(msg.payload);
        break;
      default:
        break;
    }
  }, []);

  /** Refresh the catalog, optionally re-filtering it. */
  const refresh = useCallback(
    (filters: HubQueryInput = {}) => {
      setSearch(null);
      sendHub("hub_list", filters);
    },
    [sendHub],
  );

  const runSearch = useCallback(
    (query: string, limit = 20) => sendHub("hub_search", { query, limit }),
    [sendHub],
  );

  const select = useCallback((id: string) => {
    setSelected(id);
    setInspect(null);
    setImported(null);
  }, []);

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
    [select, sendHub],
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
    [sendHub],
  );

  const importEntry = useCallback(
    (id: string) => {
      setSelected(id);
      sendHub("hub_import", {}, id);
    },
    [sendHub],
  );

  useEffect(() => {
    if (!connected) return;
    sendHub("hub_list", {});
  }, [connected, sendHub]);

  return {
    list,
    search,
    download,
    inspect,
    imported,
    selected,
    onMessage,
    refresh,
    runSearch,
    select,
    downloadEntry,
    cancelDownload,
    inspectEntry,
    importEntry,
  };
}

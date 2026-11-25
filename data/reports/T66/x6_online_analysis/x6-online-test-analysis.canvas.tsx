import {
  BarChart,
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  CollapsibleSection,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  LineChart,
  Pill,
  Row,
  Spacer,
  Stack,
  Stat,
  Swatch,
  Table,
  Text,
  UsageBar,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type TabKey = "overview" | "perception" | "planning" | "foxglove";
type TrackFilter = "all" | "moving" | "parked_tp" | "fp_flicker";

interface TrackRow {
  id: number;
  window: string;
  dur: string;
  frames: number;
  category: "moving" | "parked_tp" | "fp_flicker";
  catLabel: string;
  pos: string;
  dist: string;
  size: string;
  maxSpeed: string;
  score: string;
  note: string;
  tone: "success" | "info" | "warning" | "danger" | "neutral";
}

const BINS_20S = [
  "0-20s",
  "20-40s",
  "40-60s",
  "60-80s",
  "80-100s",
  "100-120s",
  "120-140s",
  "140-160s",
  "160-180s",
  "180-200s",
  "200-220s",
  "220-240s",
  "240-260s",
  "260-280s",
  "280-300s",
  "300-320s",
  "320-340s",
  "340-360s",
  "360-380s",
  "380-394s",
];

const MOVING_AVG_PER_BIN = [
  0.28, 0.01, 0.39, 0.88, 1.11, 1.49, 0.57, 0.98, 1.53, 0.0,
  0.0, 0.0, 0.63, 1.22, 0.01, 0.0, 0.01, 1.28, 0.1, 0.18,
];

const STATIC_TP_PER_BIN = [
  2.04, 2.18, 1.82, 1.79, 1.21, 0.59, 0.94, 0.86, 1.27, 2.0,
  2.0, 2.0, 1.86, 1.53, 2.0, 2.0, 1.53, 0.94, 1.02, 1.06,
];

const FP_FLICKER_PER_BIN = [
  0.0, 0.0, 0.0, 0.0, 0.04, 0.04, 0.0, 0.0, 0.03, 0.0,
  0.0, 0.07, 0.14, 0.62, 0.82, 0.14, 0.0, 0.14, 0.07, 0.0,
];

const HOLD_PCT_PER_BIN = [
  24.3, 25.5, 36.8, 60.8, 70.0, 78.8, 0.0, 100.0, 100.0, 21.0,
  0.0, 26.0, 58.0, 82.0, 30.5, 27.0, 0.0, 31.0, 0.0, 0.0,
];

const KEY_TRACKS: TrackRow[] = [
  {
    id: 1242,
    window: "0.0-55.1s",
    dur: "55.1s",
    frames: 545,
    category: "parked_tp",
    catLabel: "静止真车 (重卡/大巴A)",
    pos: "(-39.6, 17.4)",
    dist: "43.3 m",
    size: "14.8x2.8x4.3",
    maxSpeed: "0.0 m/s",
    score: "0.48",
    note: "左前方停放重车，后续被过往车遮挡断成 11 个 ID (1242..1345)",
    tone: "info",
  },
  {
    id: 1243,
    window: "0.0-57.7s",
    dur: "57.7s",
    frames: 578,
    category: "parked_tp",
    catLabel: "静止真车 (重卡/大巴B)",
    pos: "(21.8, 25.9)",
    dist: "33.9 m",
    size: "12.7x3.0x3.8",
    maxSpeed: "0.0 m/s",
    score: "0.86",
    note: "右前方停放，t=85.3s 启动驶离 (变为运动轨迹 ID 1283)",
    tone: "info",
  },
  {
    id: 1256,
    window: "15.1-20.0s",
    dur: "4.9s",
    frames: 38,
    category: "moving",
    catLabel: "运动真车",
    pos: "(20.6,16.6)->(30.3,19.2)",
    dist: "26.5 m",
    size: "4.6x2.0x1.8",
    maxSpeed: "2.6 m/s",
    score: "0.43",
    note: "触发首次 cross_wait (ahead)，远去消失后触发 5s lost 保持",
    tone: "success",
  },
  {
    id: 1257,
    window: "17.8-39.3s",
    dur: "21.5s",
    frames: 45,
    category: "fp_flicker",
    catLabel: "静态边缘断续框",
    pos: "(18.8, -19.7)",
    dist: "27.3 m",
    size: "4.4x1.9x1.6",
    maxSpeed: "0.0 m/s",
    score: "0.38",
    note: "右后方低分框，刚出现 age<1s (young) 时叠加触发 cross_wait",
    tone: "warning",
  },
  {
    id: 1259,
    window: "51.8-62.9s",
    dur: "11.1s",
    frames: 112,
    category: "moving",
    catLabel: "运动真车 (横穿)",
    pos: "(-38.7,-0.6)->(41.8,9.1)",
    dist: "4.3 m",
    size: "11.2x2.9x3.6",
    maxSpeed: "8.7 m/s",
    score: "0.68",
    note: "从左至右距狗 4.3m 横穿路口，全程稳定跟踪 81.1m",
    tone: "success",
  },
  {
    id: 1265,
    window: "60.2-73.5s",
    dur: "13.3s",
    frames: 134,
    category: "moving",
    catLabel: "运动真车 (横穿)",
    pos: "(-38.3,-1.2)->(41.7,7.7)",
    dist: "3.3 m",
    size: "12.8x3.0x3.8",
    maxSpeed: "8.2 m/s",
    score: "0.66",
    note: "近距 3.3m 横穿，门控正常维持 cross_wait 直至驶离后 3s",
    tone: "success",
  },
  {
    id: 1274,
    window: "78.2-89.2s",
    dur: "11.0s",
    frames: 111,
    category: "moving",
    catLabel: "运动真车 (横穿)",
    pos: "(-38.0,0.0)->(41.8,8.8)",
    dist: "4.4 m",
    size: "11.6x2.9x3.7",
    maxSpeed: "8.9 m/s",
    score: "0.68",
    note: "稳定跟踪 80.3m，门控正常触发让行",
    tone: "success",
  },
  {
    id: 1283,
    window: "85.3-108.1s",
    dur: "22.8s",
    frames: 229,
    category: "moving",
    catLabel: "运动真车 (起步驶离)",
    pos: "(20.4,25.9)->(-17.6,14.1)",
    dist: "22.6 m",
    size: "12.4x3.0x3.8",
    maxSpeed: "5.5 m/s",
    score: "0.66",
    note: "原 PosB 停放重车起步左行，跟踪连续",
    tone: "success",
  },
  {
    id: 1293,
    window: "103.5-115.3s",
    dur: "11.8s",
    frames: 119,
    category: "moving",
    catLabel: "运动真车 (横穿)",
    pos: "(-38.4,-0.5)->(44.0,8.4)",
    dist: "3.9 m",
    size: "12.0x2.9x3.7",
    maxSpeed: "8.6 m/s",
    score: "0.72",
    note: "距狗 3.9m 快速通过，跟踪完整 (82.9m)",
    tone: "success",
  },
  {
    id: 1295,
    window: "109.6-123.4s",
    dur: "13.8s",
    frames: 139,
    category: "moving",
    catLabel: "运动真车 (跟驰横穿)",
    pos: "(-16.5,9.6)->(39.7,8.8)",
    dist: "4.2 m",
    size: "12.5x3.0x3.8",
    maxSpeed: "7.7 m/s",
    score: "0.59",
    note: "与 1293 交会时短暂被前车遮挡，-16.5m 处建轨后无断轨",
    tone: "success",
  },
  {
    id: 1304,
    window: "131.6-167.4s",
    dur: "35.8s",
    frames: 355,
    category: "moving",
    catLabel: "运动转停放真车",
    pos: "(-40.3,-0.3)->(16.0,24.2)",
    dist: "4.3 m",
    size: "13.3x3.0x4.0",
    maxSpeed: "7.2 m/s",
    score: "0.64",
    note: "左来转入右前方停下 (PosC)，运动->停止状态切换准确",
    tone: "success",
  },
  {
    id: 1311,
    window: "165.0-175.9s",
    dur: "10.9s",
    frames: 109,
    category: "moving",
    catLabel: "运动真车 (右向左)",
    pos: "(36.6,13.9)->(-42.7,4.2)",
    dist: "8.6 m",
    size: "12.9x3.0x3.9",
    maxSpeed: "9.1 m/s",
    score: "0.57",
    note: "新目标重发后处于 wait (vehicle) 阶段，门控正常等待其通过",
    tone: "success",
  },
  {
    id: 1316,
    window: "170.6-264.3s",
    dur: "93.7s",
    frames: 925,
    category: "parked_tp",
    catLabel: "静止真车 (PosC)",
    pos: "(15.8, 24.2)",
    dist: "28.9 m",
    size: "13.4x3.0x4.0",
    maxSpeed: "0.0 m/s",
    score: "0.67",
    note: "1304 停稳后被过往车遮挡重开 ID (后续又裂变为 1342/1347/1364)",
    tone: "info",
  },
  {
    id: 1325,
    window: "225.7-231.4s",
    dur: "5.7s",
    frames: 15,
    category: "fp_flicker",
    catLabel: "低分静态误检/杂波",
    pos: "(10.3, 27.8)",
    dist: "29.7 m",
    size: "2.3x1.9x2.5",
    maxSpeed: "0.9 m/s",
    score: "0.39",
    note: "距 27.7m 横道路径 <30m，以 young 触发 5.2s 假让行等待",
    tone: "danger",
  },
  {
    id: 1327,
    window: "240.7-251.8s",
    dur: "11.1s",
    frames: 112,
    category: "moving",
    catLabel: "运动真车 (横穿)",
    pos: "(-38.0,-0.5)->(41.0,8.8)",
    dist: "4.3 m",
    size: "12.1x2.9x3.7",
    maxSpeed: "8.5 m/s",
    score: "0.68",
    note: "0 coast 满帧跟踪 79.6m，门控让行正常",
    tone: "success",
  },
  {
    id: 1333,
    window: "257.9-270.8s",
    dur: "12.9s",
    frames: 130,
    category: "moving",
    catLabel: "运动真车 (横穿)",
    pos: "(-38.3,-0.8)->(41.7,8.7)",
    dist: "3.5 m",
    size: "13.2x3.0x3.9",
    maxSpeed: "8.8 m/s",
    score: "0.68",
    note: "与 1338 连续两车通过，门控合并等待正常",
    tone: "success",
  },
  {
    id: 1344,
    window: "271.9-296.4s",
    dur: "24.5s",
    frames: 234,
    category: "fp_flicker",
    catLabel: "远距背景误检 (灌木/设施)",
    pos: "(-30.2, 21.5)",
    dist: "37.1 m",
    size: "2.3x1.4x2.6",
    maxSpeed: "1.1 m/s",
    score: "0.61",
    note: "左前 37m 小尺寸框，因距长横道出口 <30m 以 young 延长了等待",
    tone: "danger",
  },
  {
    id: 1349,
    window: "291.6-292.6s",
    dur: "1.0s",
    frames: 11,
    category: "fp_flicker",
    catLabel: "远距瞬态误检",
    pos: "(31.8, 33.3)",
    dist: "46.0 m",
    size: "5.9x2.4x3.2",
    maxSpeed: "1.2 m/s",
    score: "0.41",
    note: "46m 处闪烁 1.0s (2帧判动)，触发 cross_wait + 5s lost 共空等 6.1s",
    tone: "danger",
  },
  {
    id: 1352,
    window: "305.3-305.6s",
    dur: "0.3s",
    frames: 4,
    category: "fp_flicker",
    catLabel: "远距墙体/建筑闪框",
    pos: "(11.4, 34.9)",
    dist: "36.8 m",
    size: "11.2x3.0x3.7",
    maxSpeed: "0.0 m/s",
    score: "0.42",
    note: "36.8m 建筑边缘闪现 4 帧，因 young 规则导致规划空等 5.4s",
    tone: "danger",
  },
  {
    id: 1354,
    window: "339.5-359.1s",
    dur: "19.6s",
    frames: 195,
    category: "moving",
    catLabel: "运动真车 (横穿)",
    pos: "(-40.5,1.1)->(39.4,12.1)",
    dist: "4.1 m",
    size: "12.8x3.0x3.8",
    maxSpeed: "7.5 m/s",
    score: "0.57",
    note: "通过中途 (t=352.0s) 规划因边界抖动退回 none，提前解除等待",
    tone: "warning",
  },
  {
    id: 1356,
    window: "350.7-354.3s",
    dur: "3.6s",
    frames: 37,
    category: "moving",
    catLabel: "运动真车 (绕狗后方)",
    pos: "(-36.9,-17.3)->(-13.8,-14.9)",
    dist: "20.3 m",
    size: "9.8x2.8x3.4",
    maxSpeed: "9.5 m/s",
    score: "0.57",
    note: "进入狗正后方 76 度机身遮挡盲区断轨，2s 后右侧重建为 ID 1365",
    tone: "warning",
  },
  {
    id: 1368,
    window: "372.8-375.0s",
    dur: "2.2s",
    frames: 23,
    category: "moving",
    catLabel: "运动大车 (中途漏检段)",
    pos: "(38.4,23.5)->(34.8,22.9)",
    dist: "41.5 m",
    size: "10.2x2.9x3.6",
    maxSpeed: "2.9 m/s",
    score: "0.45",
    note: "右前慢速接近大车：t=375-383.6s 及 t=386.5-394s (19.4m 处) 发生漏检，仅 t=383.6-386.5s 短暂建轨为 ID 1374",
    tone: "danger",
  },
];

const PLANNING_PHASES = [
  {
    phase: "阶段 1 · 原遗留目标在横道内",
    time: "0.0 - 116.7 s (20:06:20 - 20:08:17)",
    mode: "cross (5) <-> cross_wait (6)",
    pathInfo: "exit = 2.2 m (东侧短出口)",
    behavior:
      "狗静止站在 6 号人行横道东端内部 (106.87, 209.09)。5 辆真车 (1256/1259/1265/1274/1293) 横穿时均正确触发 cross_wait (ahead)；期间 ID 1257 低分框叠加了约 7s 等待。",
    verdict: "基本正常 (原地测试)",
    tone: "success" as const,
  },
  {
    phase: "阶段 2 · 首次重发目标停掉旧任务",
    time: "116.7 - 157.0 s (20:08:17 - 20:08:57)",
    mode: "Idle / off (无横道状态)",
    pathInfo: "Foxglove 发 /goal_pose (base_link)",
    behavior:
      "第一次点 Foxglove 目标只触发了停止旧 Tracking 任务切回 Idle (DOG_FIELD_GUIDE 已知机制：需点第二次才启动新任务)，此 40.3 s 内规划不输出指令。",
    verdict: "符合交互设计",
    tone: "neutral" as const,
  },
  {
    phase: "阶段 3 · 新目标沿 32m 横道纵向穿过",
    time: "157.0 - 184.2 s (20:08:57 - 20:09:24)",
    mode: "wait (3) -> go (4, 1帧)",
    pathInfo: "exit = 27.7 m, unreleased = 0.06 m",
    behavior:
      "第二次发目标 (base_link 27.0, 4.0 -> map 80.72, 216.44)，路径沿 32m 长的 6 号横道纵向走 27.7m。狗压入口线 6cm 判作路边等车，等 1304/1310/1311/1317 全部清空后于 184.24s 放行 go。",
    verdict: "门控逻辑正常",
    tone: "success" as const,
  },
  {
    phase: "阶段 4 · 横道上定点观察 (含远距闪框空等)",
    time: "184.3 - 352.0 s (20:09:24 - 20:12:12)",
    mode: "cross (5) <-> cross_wait (6)",
    pathInfo: "exit = 27.7 m, 查整条 28.7m 路径",
    behavior:
      "影子模式下狗未动，保持在横道内。真车 1327/1333/1338/1354 均正确触发让行；但因路径长达 28.7m，远端 30-46m 处的瞬态误检框 (1325/1344/1349/1352) 命中 young (<1s, 30m内) + 5s lost_hold，造成 3 次各 5-6s 空等。",
    verdict: "存在远距闪框空等",
    tone: "warning" as const,
  },
  {
    phase: "阶段 5 · 边界微抖动触发 passed_ 死区",
    time: "352.0 - 394.1 s (20:12:12 - 20:12:54)",
    mode: "none (0), unreleased = 1",
    pathInfo: "crosswalk -1 (门控完全旁路)",
    behavior:
      "狗站在多边形边界内侧仅 0-7cm。t=352.0s 定位微抖几厘米出界 3 帧，因 passed_ 已记录 crosswalk 0，状态机切回 none；0.15s 后抖回界内 7mm，因 depth<=1.5m 置 inside=-1，而下一步又因 passed_.count(0) 跳过该横道，导致最后 42s 门控永久失效 (期间 1354 仍在横穿)。",
    verdict: "发现状态机边界 Bug",
    tone: "danger" as const,
  },
];

export default function X6OnlineTestCanvas() {
  const theme = useHostTheme();
  const dispatch = useCanvasAction();
  const [tab, setTab] = useCanvasState<TabKey>("tab", "overview");
  const [trackFilter, setTrackFilter] = useCanvasState<TrackFilter>("trackFilter", "all");

  const filteredTracks = KEY_TRACKS.filter((r) =>
    trackFilter === "all" ? true : r.category === trackFilter
  );

  return (
    <Stack gap={20} style={{ padding: 20 }}>
      <Stack gap={6}>
        <Row gap={8} align="center" wrap>
          <Text size="small" tone="tertiary">
            20261115_x6_all_1_0.db3 · 8号狗在线影子测试 · 站点地图 ad68517c (6号路口 / crosswalk 0)
          </Text>
          <Spacer />
          <Button
            variant="secondary"
            onClick={() =>
              dispatch({
                type: "openFile",
                path: "d20car_predict/CenterPoint/Nav_D20/planning/script/planning_live.json",
              })
            }
          >
            打开 planning_live.json
          </Button>
          <Button
            variant="secondary"
            onClick={() =>
              dispatch({
                type: "openFile",
                path: "d20car_predict/CenterPoint/Nav_D20/planning/src/runnable/mission/tracking/crosswalk_yield.cc",
                selection: { startLineNumber: 195, endLineNumber: 256 },
              })
            }
          >
            定位 crosswalk_yield.cc
          </Button>
        </Row>
        <H1>x6 路口 3D 点云车辆识别与让行规划在线测试分析</H1>
        <Text tone="secondary">
          时长 394.1 s (20:06:20–20:12:54)，狗全程静止站在 6 号人行横道东侧入口线内侧 (位移 0.24 m，影子模式开启，RTK 100% 固定解)。
          共采集激光雷达 3,941 帧 (10.0 Hz，0 丢帧)、四路环视鱼眼各 9,851 帧、规划周期 7,881 次 (20.0 Hz)。
        </Text>
      </Stack>

      <Row gap={8} wrap>
        <Pill active={tab === "overview"} onClick={() => setTab("overview")}>
          总览与核心结论
        </Pill>
        <Pill active={tab === "perception"} onClick={() => setTab("perception")}>
          3D 点云感知 (漏检 / 误检)
        </Pill>
        <Pill active={tab === "planning"} onClick={() => setTab("planning")}>
          路口让车规划状态机
        </Pill>
        <Pill active={tab === "foxglove"} onClick={() => setTab("foxglove")}>
          Foxglove 回放与 Layout 指南
        </Pill>
      </Row>

      <Divider />

      {tab === "overview" && (
        <Stack gap={20}>
          <Grid columns={4} gap={12}>
            <Stat
              value="93.3% (14/15)"
              label="主路口 35m 内过往真车召回 (0 完全漏检，1 例近距断续)"
              tone="success"
            />
            <Stat
              value="5.4% 帧占比"
              label="误检轨迹框帧数占比 (468 / 8,603 框·帧，多在 34–52m 边缘)"
              tone="warning"
            />
            <Stat
              value="26.5 ms"
              label="Orin 端到端感知耗时中位数 (p99 39.7 ms，0 丢帧)"
              tone="info"
            />
            <Stat
              value="1 处边界 Bug"
              label="规划前期让车正常，但 352s 因压线抖动掉入 passed_ 死区"
              tone="danger"
            />
          </Grid>

          <Callout tone="warning" title="三句话回答核心问题">
            1. 3D 识别漏检与误检：主干道横穿真车漏检很低（14 辆近距横穿车 100% 稳定检出且无断轨；仅末尾 1 辆慢速大车在右前 19–35m 侧向接近时断续漏检，外加狗正后方 76 度物理遮挡盲区）；帧级误检不高（占总输出框 5.4%），但 30–46m 边缘存在 19 条瞬态静态假框（0.2–4s）及静止重车被遮挡后的 ID 重建。
            2. 规划让车状态：C++ 门控对真实过往车辆的响应正常（横穿车均准确触发 wait / cross_wait 并在清空 3s 后放行），但暴露两个现场问题——一是 27.7m 长横道放大了远端瞬态假框的 young+lost 空等（3 次各空等 5–6s），二是狗站在入口线内侧几厘米处因定位微抖触发了 crosswalk_yield.cc 的 passed_ 死区 Bug，导致最后 42s 门控被旁路。
            3. Foxglove 回放：桌面版可直接打开 .db3 看四路视频、3D 路网与 /planning/debug/crosswalk 里的 3D 车框，但自定义消息 (/perception/lidar_3d/objects 等) 需转 mcap 才能解码，且原始激光点云因缺 /tf_static 需切到 hesai_lidar 坐标系查看；主布局在 Nav_D20/planning/script/planning_live.json。
          </Callout>

          <Stack gap={8}>
            <H2>全时段（每 20 秒窗口）车辆跟踪数与规划驻留等待率</H2>
            <Text size="small" tone="tertiary">
              横轴：包内相对时间窗口 (s) · 纵轴：每帧平均目标数 (个/帧) · 来源：/perception/lidar_3d/objects (3,941 帧，20:06:20–20:12:54)
            </Text>
            <BarChart
              categories={BINS_20S}
              series={[
                { name: "运动真车 (个/帧)", data: MOVING_AVG_PER_BIN, tone: "success" },
                { name: "停放真车 (个/帧)", data: STATIC_TP_PER_BIN, tone: "info" },
                { name: "远距闪烁/误检框 (个/帧)", data: FP_FLICKER_PER_BIN, tone: "danger" },
              ]}
              stacked
              height={220}
            />
          </Stack>

          <Stack gap={8}>
            <H2>规划让行驻留比（crosswalk_hold = 1 占比）随时间变化</H2>
            <Text size="small" tone="tertiary">
              横轴：包内相对时间窗口 (s) · 纵轴：门控驻留制动周期占比 (%) · 注：120-140s 为任务切换 Idle，352s 后因边界死区掉回 0% · 来源：/planning/debug/plotjuggler/string
            </Text>
            <LineChart
              categories={BINS_20S}
              series={[
                { name: "规划让行 Hold 占比 (%)", data: HOLD_PCT_PER_BIN, tone: "warning" },
              ]}
              valueSuffix="%"
              yMin={0}
              yMax={100}
              fill
              height={190}
            />
          </Stack>
        </Stack>
      )}

      {tab === "perception" && (
        <Stack gap={20}>
          <Stack gap={8}>
            <H2>1. 在线节点运行健康度与算力指标</H2>
            <Table
              headers={["指标项", "实测数值", "设计预期 / 阈值", "结论"]}
              rows={[
                ["激光与目标帧率", "3,941 帧 / 394.1 s (10.00 Hz, max dt 101.5 ms)", "10 Hz 稳定无断流", "零丢帧 (skipped = 0)"],
                ["单帧总耗时 total_ms", "p50 = 26.5 ms · p90 = 32.5 ms · p99 = 39.7 ms", "< 50 ms (p99 < 100 ms)", "实时裕量充足"],
                ["阶段耗时拆分 (p50)", "预处理 10.9 ms + TRT GPU 13.2 ms + 后处理 2.0 ms + 跟踪预测 0.1 ms", "Orin FP16 动态 PFN", "GPU 推理极稳"],
                ["底层过滤框 -> 确认航迹", "每帧中位 31.5 个低分候选 (>=0.2) -> 输出中位 2.0 条航迹 (>=0.4, min_hits=3)", "抑制低机位地面/灌木噪点", "过滤掉约 93% 底层噪点"],
                ["定位与时间同步", "INSPVAX 100% RTK 固定解；0 帧与里程计时间差 > 0.3s", "crosswalk_stale == 0", "外参时间链完全对齐"],
              ]}
              rowTone={["success", "success", "success", "info", "success"]}
            />
          </Stack>

          <Stack gap={10}>
            <H2>2. 漏检（FN）与误检（FP）定量拆解</H2>
            <UsageBar
              total={8603}
              topLeftLabel="全部 8,603 个在线输出框·帧的真伪构成"
              topRightLabel="真车框·帧占 94.6% (8,135) / 疑似误检占 5.4% (468)"
              segments={[
                { id: "parked_tp", value: 5913, color: "blue" },
                { id: "moving_tp", value: 2222, color: "green" },
                { id: "fp_clutter", value: 468, color: "red" },
              ]}
            />
            <Grid columns={2} gap={14}>
              <Card>
                <CardHeader trailing={<Pill size="sm">低漏检 · 2 类已知边界</Pill>}>
                  漏检分析 (Missed Detections / FN)
                </CardHeader>
                <CardBody>
                  <Stack gap={8}>
                    <Text>
                      通过「3D 激光静态背景体素差分聚类 + 四路环视鱼眼逐段核对」交叉验证：
                    </Text>
                    <Text size="small" tone="secondary">
                      1. 主路口横穿车（距狗 3–35 m）：共 14 辆次大车/轿车从狗正前方横穿或转弯（如 ID 1259, 1265, 1274, 1293, 1295, 1304, 1311, 1327, 1333, 1338, 1354 等），检出率 100%，单车连续跟踪最高 82.9 m，无一完全漏检。
                    </Text>
                    <Text size="small" tone="secondary">
                      2. 近距慢速大车侧向漏检（1 例）：t = 372.8–394.1 s 右前方一辆约 10.4m 大车从 41.5m 慢速驶近至 19.4m（激光前景达 947 点），仅在 372.8–375.0s (ID 1368) 和 383.6–386.5s (ID 1374) 出框，375–383s 与 386.5–394s 分数跌破 0.40 断轨。正对应 T39 复核指出的「nusc_pp 在低机位对 20m 内大货车召回偏弱 (0.833)」。
                    </Text>
                    <Text size="small" tone="secondary">
                      3. 硬件盲区与配置范围外目标：狗正后方 |方位角| &gt; 140° 存在约 80° 机身遮挡盲区，t = 354.3–356.3s 绕狗后方驶过的车 (ID 1356 -&gt; 1365) 在盲区内必然断轨 2.0s；狗车尾站立的操作员及 24m 处 1.3m 宽骑行/行人不输出框，是因为配置了 tracker.classes: [&quot;vehicle&quot;] 只跟踪机动车辆。
                    </Text>
                  </Stack>
                </CardBody>
              </Card>

              <Card>
                <CardHeader trailing={<Pill size="sm">帧占比 5.4% · ID 碎片偏多</Pill>}>
                  误检与轨迹分裂分析 (False Positives / ID Split)
                </CardHeader>
                <CardBody>
                  <Stack gap={8}>
                    <Text>
                      全包共生成 61 个唯一航迹 ID（8,603 框·帧），按航迹 ID 看似乎多，但拆解后主要由两类现象构成：
                    </Text>
                    <Text size="small" tone="secondary">
                      1. 远距离建筑/灌木边缘瞬态假框（约 19 个 ID，仅 423 框·帧 = 4.9%）：集中在 34–52 m 处（如 (-11.3, 36.4)、(11.4, 35.3) 厂房立面边缘、(-30.2, 21.5) 绿化带设施），置信度踩在 0.39–0.52 门限边缘，大多持续 0.2–3.5 s（3–18 帧）。20 m 近场内零静态假框。
                    </Text>
                    <Text size="small" tone="secondary">
                      2. 停放重车被遮挡后的 ID 裂变（3 辆真车占了 18 个 ID，5,913 框·帧）：左前 (-39.6, 17.4) 停放车被近处横穿大车遮挡超过 max_age = 1.5s 后重开新号，裂变成 11 个 ID；右前 PosB/PosC 停放重车裂变成 7 个 ID。感知本身位置极稳，但每次重开新 ID 时 age 从 0 计，会触发下游门控的 young 规则。
                    </Text>
                    <Text size="small" tone="secondary">
                      3. 大车局部框重叠（1 例）：t = 385s 右前大车同时出现静态长框与局部低分运动小框 (ID 1374, score 0.36)。
                    </Text>
                  </Stack>
                </CardBody>
              </Card>
            </Grid>
          </Stack>

          <Stack gap={8}>
            <Row gap={8} align="center" wrap>
              <H2>3. 典型航迹明细表（支持分类筛选）</H2>
              <Spacer />
              <Pill active={trackFilter === "all"} onClick={() => setTrackFilter("all")}>
                全部代表航迹 ({KEY_TRACKS.length})
              </Pill>
              <Pill active={trackFilter === "moving"} onClick={() => setTrackFilter("moving")}>
                运动真车
              </Pill>
              <Pill active={trackFilter === "parked_tp"} onClick={() => setTrackFilter("parked_tp")}>
                停放真车 (含遮挡重开)
              </Pill>
              <Pill active={trackFilter === "fp_flicker"} onClick={() => setTrackFilter("fp_flicker")}>
                疑似误检 / 闪烁框
              </Pill>
            </Row>
            <Table
              headers={["ID", "时间窗", "持续/帧数", "类别判定", "检测系坐标 (m)", "距离", "尺寸 LxWxH", "峰值速度", "均分", "现象与对规划影响"]}
              rows={filteredTracks.map((r) => [
                String(r.id),
                r.window,
                `${r.dur} (${r.frames})`,
                r.catLabel,
                r.pos,
                r.dist,
                r.size,
                r.maxSpeed,
                r.score,
                r.note,
              ])}
              rowTone={filteredTracks.map((r) => r.tone)}
              striped
            />
          </Stack>
        </Stack>
      )}

      {tab === "planning" && (
        <Stack gap={20}>
          <Stack gap={8}>
            <H2>1. 规划周期状态分布（7,881 个周期，20 Hz）</H2>
            <UsageBar
              total={7881}
              topLeftLabel="crosswalk_mode 全周期占比"
              topRightLabel="cross 44.9% · cross_wait 27.3% · none 10.7% · Idle 10.2% · wait 6.9%"
              segments={[
                { id: "cross", value: 3537, color: "green" },
                { id: "cross_wait", value: 2151, color: "orange" },
                { id: "none", value: 843, color: "red" },
                { id: "idle", value: 807, color: "gray" },
                { id: "wait", value: 542, color: "yellow" },
                { id: "go", value: 1, color: "cyan" },
              ]}
            />
          </Stack>

          <Stack gap={8}>
            <H2>2. 五阶段时间线还原：规划在路口让车到底正不正常？</H2>
            <Table
              headers={["阶段", "时间段", "横道状态机", "路径与几何", "详细过程与感知联动表现", "评估结论"]}
              rows={PLANNING_PHASES.map((p) => [
                p.phase,
                p.time,
                p.mode,
                p.pathInfo,
                p.behavior,
                p.verdict,
              ])}
              rowTone={PLANNING_PHASES.map((p) => p.tone)}
              striped
            />
          </Stack>

          <Stack gap={10}>
            <H2>3. 本次在线测试暴露的 3 个规划/测试耦合问题（按严重度排序）</H2>

            <CollapsibleSection
              title="问题 A [高优 Bug]：狗压入口线站立 + 定位厘米级抖动，触发 passed_ 集合死区导致最后 42s 不再让车"
              leading={<Swatch color="red" />}
              trailing={<Text size="small" tone="tertiary">crosswalk_yield.cc:197-254</Text>}
              defaultOpen
            >
              <Stack gap={6} style={{ paddingTop: 6 }}>
                <Text size="small" tone="secondary">
                  复现链路：狗站在 (106.87, 209.09)，位于 6 号人行横道东端边线内侧仅 0.3–7.2 cm。t = 184.24s 门控放行进入 cross 时执行了 passed_.insert(0)。影子模式下狗未走动，t = 351.99s 定位微抖几厘米使狗中心短暂落到多边形外 3 帧 (0.15s)，因 passed_.count(0) 为真，状态机重置为 mode_ = kNone, current_ = -1。
                </Text>
                <Text size="small" tone="secondary">
                  0.15s 后定位抖回多边形内 6.9 mm，此时 current_ == -1 触发 unreleased = true；因 depth (0.007m) &lt;= max_unreleased_depth (1.5m)，第 209 行将 inside 置为 -1 企图按「路边未过横道」重走第 241 行查找，但第 242 行 if (passed_.count(0)) continue 直接跳过了 0 号横道！结果狗明明还剩 27.7m 横道没过，状态却永久锁死在 none (hold=0)，此时横穿的真车 ID 1354 被完全忽略。
                </Text>
              </Stack>
            </CollapsibleSection>

            <CollapsibleSection
              title="问题 B [中优]：32m 超宽斜交横道 + 远距瞬态假框 (young < 1s) + 5s lost_hold 叠加造成空等"
              leading={<Swatch color="orange" />}
              trailing={<Text size="small" tone="tertiary">3 次空等 · 每次 5.2-6.1s</Text>}
              defaultOpen
            >
              <Stack gap={6} style={{ paddingTop: 6 }}>
                <Text size="small" tone="secondary">
                  6 号人行横道横跨整个西岔口（长 32.2m），新目标使全局路径在横道内纵向延伸 27.7m（加上 check_beyond_exit 共查 28.7m）。门控规则对路径 30m 内的任何新航迹 (age &lt; 1.0s) 一律判 young 冲突；一旦 35–46m 处的建筑边缘闪现 0.3s 假框（如 ID 1352 仅 4 帧、ID 1349 仅 1.0s）随即消失，门控立刻转入 lost_hold 再硬等 5.0s，导致无车时原地多停 5.4–6.1s；此外停放重车被真车遮挡后重开新 ID (如 1347) 也会附带触发 young。
                </Text>
              </Stack>
            </CollapsibleSection>

            <CollapsibleSection
              title="问题 C [工具链隐患]：Foxglove 在 base_link 系发目标，nav_cw_record.py 未转 map 系写错 replay goal"
              leading={<Swatch color="yellow" />}
              trailing={<Text size="small" tone="tertiary">record_log.txt 末尾</Text>}
            >
              <Stack gap={6} style={{ paddingTop: 6 }}>
                <Text size="small" tone="secondary">
                  Foxglove 3D 面板跟随 base_link 时发出的 /goal_pose header.frame_id 为 base_link (27.004, 4.026)。cw-planning 内部用 TF 正确转成了 map 系 (80.72, 216.44)，但录包脚本 nav_cw_record.py 直接把 (27.004, 4.026) 当作 map 坐标算成「距狗 220.1m」并写入 replay goal: 27.004 4.026。后续若直接跑 replay_dog_bag.sh 回灌此包，终点会偏出两百多米，需手动传终点 80.72 216.44。
                </Text>
              </Stack>
            </CollapsibleSection>
          </Stack>
        </Stack>
      )}

      {tab === "foxglove" && (
        <Stack gap={20}>
          <Callout tone="info" title="可以直接用 Foxglove 打开 20261115_x6_all_1_0.db3 吗？">
            可以打开看绝大部分可视化内容（四路环视视频、路网、局部栅格、路径、以及 /planning/debug/crosswalk 里已转到 map 系的 3D 车框和让行状态），但直接开 .db3 有 3 个限制：自定义消息无内置 Schema 无法解码、原始激光点云缺静态 TF、以及 14.8 GB 单文件加载较慢。按你的查看目的选下面三种方式之一即可。
          </Callout>

          <Stack gap={8}>
            <H2>1. 三种回放方式对比（能看到什么 / 看不到什么）</H2>
            <Table
              headers={["回放方式", "能正常显示的内容", "看不了 / 缺失的内容", "适用场景"]}
              rows={[
                [
                  "方式 A：Foxglove 直接打开原始 20261115_x6_all_1_0.db3",
                  "四路鱼眼相机视频 (CompressedVideo)、/planning/debug/crosswalk (含人行横道变色框 + map 系 3D 车框 + 速度箭头 + 状态字)、代价地图、局部栅格、全局/局部路径、/lightning/odom、/nav_cw/shadow/cmd_vel、plotjuggler 文本",
                  "1) 自定义类型无法解码：/perception/lidar_3d/objects、/nav_cw/shadow/mc/motion_control、/rte/rdap/rtk/inspvax；2) 原始点云 /robot/topic/rdap/top_point_cloud2 因包内无 /tf_static (缺 base_link->hesai_lidar)，在 map 系画不出，需把 3D 面板 Display Frame 临时改成 hesai_lidar 单看点云",
                  "零转换直接看四路相机实况 + 规划 3D 车框与让行状态",
                ],
                [
                  "方式 B：bag_to_mcap.py 转成 .mcap 后打开",
                  "补齐仓库所有自定义 ROS 消息 Schema：/perception/lidar_3d/objects 原始目标属性表、/nav_cw/shadow/mc/motion_control 曲线、INSPVAX 全部可读",
                  "仓库 typestore 默认未注册 foxglove_msgs/CompressedVideo 与 hesai_ros_driver/LossPacket，转出的 mcap 会提示 LEFT OUT 这 5 个话题（无相机视频）；同样缺 hesai_lidar 静态 TF",
                  "配合 planning_live.json 完整核对感知原始字段与规划调试数",
                ],
                [
                  "方式 C：再跑 planning_mcap_viz.py 生成 planning_viz.mcap",
                  "生成轻量防闪烁精排包：自带完整 ad68517c 站点路网、带编号的 5 条人行横道、拍平分层无 Z-fighting 闪烁的代价地图与感知栅格、结构化 /viz/state 与 /viz/events",
                  "不含原始点云与相机视频（纯 2.5D 规划全局复盘视图）",
                  "最流畅地拖动时间轴复盘整段规划状态机与轨迹",
                ],
              ]}
              rowTone={["info", "success", "success"]}
              striped
            />
          </Stack>

          <Stack gap={8}>
            <H2>2. 仓库内 Layout 布局文件位置清单</H2>
            <Table
              headers={["Layout 文件路径 (相对 d20car_predict/CenterPoint/)", "匹配数据源", "面板结构与用途", "快捷操作"]}
              rows={[
                [
                  "Nav_D20/planning/script/planning_live.json",
                  "原始 .db3 或 bag_to_mcap 转出的 .mcap (首选推荐)",
                  "左 3D (跟 base_link，含 crosswalk 车框/栅格/路径) + 右上速度/角速度曲线 (含 shadow 指令) + 右下三标签页 (规划数 / 诊断 / 感知目标)。看相机只需点右上 Add Panel -> Image 选 front/left/right/rear_sensor_camera",
                  "首选",
                ],
                [
                  "Nav_D20/planning/script/planning_viz.json",
                  "planning_mcap_viz.py 生成的 planning_viz.mcap",
                  "防闪烁精排 3D (/viz/* 分层话题) + 事件日志 + 状态曲线，适合回放复盘",
                  "精排复盘",
                ],
                [
                  "Nav_D20/perception_combo_ros/node_src/lidar_3d_detection/foxglove/lidar_3d_detection.json",
                  "开了 debug perception 的包 (本包不适用)",
                  "跟随 camera_init 系看 /perception/lidar_3d/{cloud,tracks,predictions}；本包未开感知调试故未录这三个话题",
                  "仅调试包用",
                ],
                [
                  "Nav_D20/planning/script/planning_debug.json · planning_io.json · loc_viz.json",
                  "模块单项调试包",
                  "纯单窗 3D 或定位轨迹比对，不含人行横道侧栏与影子指令曲线",
                  "单项调试",
                ],
              ]}
              rowTone={["success", "info", "warning", "neutral"]}
            />
            <Row gap={8} wrap>
              <Button
                variant="primary"
                onClick={() =>
                  dispatch({
                    type: "openFile",
                    path: "d20car_predict/CenterPoint/Nav_D20/planning/script/planning_live.json",
                  })
                }
              >
                在编辑器中打开 planning_live.json
              </Button>
              <Button
                variant="secondary"
                onClick={() =>
                  dispatch({
                    type: "openFile",
                    path: "d20car_predict/CenterPoint/Nav_D20/planning/script/planning_viz.json",
                  })
                }
              >
                在编辑器中打开 planning_viz.json
              </Button>
            </Row>
          </Stack>
        </Stack>
      )}
    </Stack>
  );
}

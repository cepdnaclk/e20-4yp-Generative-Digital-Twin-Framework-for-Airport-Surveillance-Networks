/* ==========================================================================
 * Intent-Driven Digital Twin Orchestrator — Airport CCTV Surveillance
 *
 * Deterministic ns-3 simulation.
 * Reads  one JSON input  → runs simulation → writes one JSON output.
 *
 * Usage:
 *   ./ns3 run "scratch/simulation --input=input.json --output=output.json"
 *
 * Dependencies:
 *   - nlohmann/json (json.hpp) placed in scratch/ alongside this file.
 *
 * Build:
 *   cp simulation.cc json.hpp  <ns3>/scratch/
 *   cd <ns3> && ./ns3 build
 * ==========================================================================*/

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/traffic-control-module.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "json.hpp"

using namespace ns3;
using json = nlohmann::json;

NS_LOG_COMPONENT_DEFINE("AirportCctvDtSimulation");

// ============================================================================
//  Constants
// ============================================================================
static const uint32_t NUM_ZONES        = 6;
static const uint32_t CAMERAS_PER_ZONE = 10;
static const uint32_t TOTAL_CAMERAS    = NUM_ZONES * CAMERAS_PER_ZONE;
static const uint32_t NUM_EDGE_SERVERS = 3;
static const uint16_t SINK_PORT_BASE   = 5000;

// ============================================================================
//  Per-zone configuration
// ============================================================================
struct ZoneConfig
{
    uint32_t    zoneId;
    double      bitrateMbps;
    uint32_t    priorityClass;       // 1 = high, 2 = medium, 3 = low
    std::string modelType;           // "lightweight", "medium", "heavy"
    std::string processingLocation;  // "edge1", "edge2", "edge3", "cloud"
};

// ============================================================================
//  Full scenario input
// ============================================================================
struct ScenarioInput
{
    std::string             scenarioId;
    std::vector<ZoneConfig> zones;
    double                  backgroundTrafficMbps;
    double                  simulationTime;
};

// ============================================================================
//  Helpers
// ============================================================================

static ScenarioInput
ReadInputJson(const std::string& path)
{
    std::ifstream ifs(path);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Cannot open input file: " + path);
    }

    json j;
    ifs >> j;

    ScenarioInput input;
    input.scenarioId            = j.at("scenario_id").get<std::string>();
    input.backgroundTrafficMbps = j.value("background_traffic_mbps", 0.0);
    input.simulationTime        = j.value("simulation_time", 30.0);

    for (auto& zj : j.at("zones"))
    {
        ZoneConfig zc;
        zc.zoneId             = zj.at("zone_id").get<uint32_t>();
        zc.bitrateMbps        = zj.at("bitrate_mbps").get<double>();
        zc.priorityClass      = zj.at("priority_class").get<uint32_t>();
        zc.modelType          = zj.at("model_type").get<std::string>();
        zc.processingLocation = zj.at("processing_location").get<std::string>();
        input.zones.push_back(zc);
    }

    std::sort(input.zones.begin(), input.zones.end(),
              [](const ZoneConfig& a, const ZoneConfig& b)
              { return a.zoneId < b.zoneId; });

    if (input.zones.size() != NUM_ZONES)
    {
        throw std::runtime_error("Expected exactly 6 zones in input JSON.");
    }

    return input;
}

static double
ModelTypeToDelayMs(const std::string& modelType)
{
    if (modelType == "lightweight") return 5.0;
    if (modelType == "medium")      return 10.0;
    if (modelType == "heavy")       return 20.0;
    return 10.0;
}

static uint8_t
PriorityToTos(uint32_t priorityClass)
{
    switch (priorityClass)
    {
        case 1:  return 0xB8; // EF
        case 2:  return 0x48; // AF21
        case 3:
        default: return 0x00; // BE
    }
}

static Ipv4Address
ResolveDestination(const std::string& loc,
                   const std::vector<Ipv4Address>& edgeAddrs,
                   const Ipv4Address& cloudAddr)
{
    if (loc == "edge1") return edgeAddrs[0];
    if (loc == "edge2") return edgeAddrs[1];
    if (loc == "edge3") return edgeAddrs[2];
    if (loc == "cloud") return cloudAddr;
    return edgeAddrs[0];
}

class SubnetAllocator
{
  public:
    SubnetAllocator() : m_b(0), m_c(0) {}

    Ipv4AddressHelper Next()
    {
        std::ostringstream base;
        base << "10." << m_b << "." << m_c << ".0";
        Ipv4AddressHelper h;
        h.SetBase(base.str().c_str(), "255.255.255.0");
        if (++m_c > 255) { m_c = 0; ++m_b; }
        return h;
    }

  private:
    uint32_t m_b, m_c;
};

/* ---- Install QoS with SMALL queue sizes -------------------------------- */
static void
InstallQos(NetDeviceContainer& devs, uint32_t maxPackets)
{
    TrafficControlHelper tch;
    tch.SetRootQueueDisc("ns3::PfifoFastQueueDisc",
                          "MaxSize", StringValue(std::to_string(maxPackets) + "p"));
    tch.Install(devs);
}

// ============================================================================
//  Main
// ============================================================================
int
main(int argc, char* argv[])
{
    std::string inputFile  = "input.json";
    std::string outputFile = "output.json";

    CommandLine cmd;
    cmd.AddValue("input",  "Input JSON file path",  inputFile);
    cmd.AddValue("output", "Output JSON file path",  outputFile);
    cmd.Parse(argc, argv);

    ScenarioInput input = ReadInputJson(inputFile);
    std::cout << "[INFO] Scenario : " << input.scenarioId << std::endl;
    std::cout << "[INFO] Sim time : " << input.simulationTime << " s" << std::endl;

    // ==================================================================
    //  1.  CREATE NODES
    // ==================================================================
    NodeContainer cameras;      cameras.Create(TOTAL_CAMERAS);
    NodeContainer zoneSwitches;  zoneSwitches.Create(NUM_ZONES);
    NodeContainer coreRouter;    coreRouter.Create(1);
    NodeContainer edgeServers;   edgeServers.Create(NUM_EDGE_SERVERS);
    NodeContainer cloudServer;   cloudServer.Create(1);

    NodeContainer allNodes;
    allNodes.Add(cameras);
    allNodes.Add(zoneSwitches);
    allNodes.Add(coreRouter);
    allNodes.Add(edgeServers);
    allNodes.Add(cloudServer);

    InternetStackHelper internet;
    internet.Install(allNodes);

    SubnetAllocator subnets;

    // ==================================================================
    //  2.  Camera → Zone Switch  (30 Mbps per link, 1 ms)
    //
    //  Each camera gets a 30 Mbps link.
    //  A camera sending 25 Mbps uses 83% of its link → some queuing.
    //  A camera sending 8 Mbps uses 27% → no issue.
    // ==================================================================
    PointToPointHelper camLink;
    camLink.SetDeviceAttribute("DataRate", StringValue("30Mbps"));
    camLink.SetChannelAttribute("Delay", StringValue("1ms"));
    camLink.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("20p"));

    std::vector<Ipv4Address> cameraAddrs(TOTAL_CAMERAS);

    for (uint32_t z = 0; z < NUM_ZONES; ++z)
    {
        for (uint32_t c = 0; c < CAMERAS_PER_ZONE; ++c)
        {
            uint32_t idx = z * CAMERAS_PER_ZONE + c;
            NetDeviceContainer devs = camLink.Install(cameras.Get(idx),
                                                       zoneSwitches.Get(z));
            InstallQos(devs, 50);
            Ipv4AddressHelper ah = subnets.Next();
            Ipv4InterfaceContainer ifc = ah.Assign(devs);
            cameraAddrs[idx] = ifc.GetAddress(0);
        }
    }

    // ==================================================================
    //  3.  Zone Switch → Core Router  (100 Mbps, 2–5 ms)
    //
    //  10 cameras × 25 Mbps = 250 Mbps demand on a 100 Mbps link
    //  → guaranteed packet loss and queuing delay for high-bitrate zones
    //  10 cameras × 10 Mbps = 100 Mbps → right at capacity
    //  10 cameras × 8  Mbps = 80  Mbps → fits comfortably
    //
    //  Queue: 50 packets → fills in ~0.5 ms at saturation → drops quickly
    // ==================================================================
    for (uint32_t z = 0; z < NUM_ZONES; ++z)
    {
        PointToPointHelper swLink;
        swLink.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
        swLink.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("30p"));

        uint32_t delayMs = 2 + (z % 4);
        std::ostringstream ds;
        ds << delayMs << "ms";
        swLink.SetChannelAttribute("Delay", StringValue(ds.str()));

        NetDeviceContainer devs = swLink.Install(zoneSwitches.Get(z),
                                                  coreRouter.Get(0));
        InstallQos(devs, 50);
        Ipv4AddressHelper ah = subnets.Next();
        ah.Assign(devs);
    }

    // ==================================================================
    //  4.  Core → Edge Servers  (150 Mbps, 5 ms)
    //
    //  If 3 zones target same edge: 3 × 100 Mbps (post-switch-bottleneck)
    //  = 300 Mbps on 150 Mbps → heavy congestion
    //  If 1 zone targets edge: ~100 Mbps on 150 Mbps → near capacity
    //
    //  Queue: 40 packets
    // ==================================================================
    PointToPointHelper edgeLink;
    edgeLink.SetDeviceAttribute("DataRate", StringValue("150Mbps"));
    edgeLink.SetChannelAttribute("Delay", StringValue("5ms"));
    edgeLink.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("40p"));

    std::vector<Ipv4Address> edgeAddrs(NUM_EDGE_SERVERS);

    for (uint32_t e = 0; e < NUM_EDGE_SERVERS; ++e)
    {
        NetDeviceContainer devs = edgeLink.Install(coreRouter.Get(0),
                                                    edgeServers.Get(e));
        InstallQos(devs, 60);
        Ipv4AddressHelper ah = subnets.Next();
        Ipv4InterfaceContainer ifc = ah.Assign(devs);
        edgeAddrs[e] = ifc.GetAddress(1);
    }

    // ==================================================================
    //  5.  Core → Cloud  (80 Mbps, 25 ms)
    //
    //  WAN link — very limited.
    //  1 zone × 10 Mbps × 10 cams = 100 Mbps → already over 80 Mbps
    //  + background traffic → severe congestion
    //
    //  Queue: 30 packets → very small buffer, fast drops
    // ==================================================================
    PointToPointHelper cloudLink;
    cloudLink.SetDeviceAttribute("DataRate", StringValue("80Mbps"));
    cloudLink.SetChannelAttribute("Delay", StringValue("25ms"));
    cloudLink.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("30p"));

    NetDeviceContainer cloudDevs = cloudLink.Install(coreRouter.Get(0),
                                                      cloudServer.Get(0));
    InstallQos(cloudDevs, 40);
    Ipv4AddressHelper cloudAh = subnets.Next();
    Ipv4InterfaceContainer cloudIfc = cloudAh.Assign(cloudDevs);
    Ipv4Address cloudAddr = cloudIfc.GetAddress(1);

    // ==================================================================
    //  6.  Routing
    // ==================================================================
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // ==================================================================
    //  7.  Sink applications
    // ==================================================================
    for (uint32_t cam = 0; cam < TOTAL_CAMERAS; ++cam)
    {
        uint16_t port = SINK_PORT_BASE + cam;
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                               InetSocketAddress(Ipv4Address::GetAny(), port));

        for (uint32_t e = 0; e < NUM_EDGE_SERVERS; ++e)
        {
            ApplicationContainer app = sink.Install(edgeServers.Get(e));
            app.Start(Seconds(0.0));
            app.Stop(Seconds(input.simulationTime));
        }

        ApplicationContainer app = sink.Install(cloudServer.Get(0));
        app.Start(Seconds(0.0));
        app.Stop(Seconds(input.simulationTime));
    }

    // ==================================================================
    //  8.  Camera streaming applications
    // ==================================================================
    uint32_t packetSize = 1316;

    for (uint32_t z = 0; z < NUM_ZONES; ++z)
    {
        const ZoneConfig& zc = input.zones[z];
        double bitrateBps    = zc.bitrateMbps * 1e6;
        uint8_t tos          = PriorityToTos(zc.priorityClass);
        Ipv4Address dst      = ResolveDestination(zc.processingLocation,
                                                   edgeAddrs, cloudAddr);

        for (uint32_t c = 0; c < CAMERAS_PER_ZONE; ++c)
        {
            uint32_t camIdx = z * CAMERAS_PER_ZONE + c;
            uint16_t port   = SINK_PORT_BASE + camIdx;

            InetSocketAddress remote(dst, port);
            remote.SetTos(tos);

            OnOffHelper onoff("ns3::UdpSocketFactory", remote);
            onoff.SetConstantRate(DataRate(static_cast<uint64_t>(bitrateBps)),
                                  packetSize);
            onoff.SetAttribute("OnTime",
                StringValue("ns3::ConstantRandomVariable[Constant=1]"));
            onoff.SetAttribute("OffTime",
                StringValue("ns3::ConstantRandomVariable[Constant=0]"));

            ApplicationContainer app = onoff.Install(cameras.Get(camIdx));
            double startOffset = 0.5 + 0.001 * camIdx;
            app.Start(Seconds(startOffset));
            app.Stop(Seconds(input.simulationTime));
        }
    }

    // ==================================================================
    //  9.  Background traffic (best-effort UDP, cloud → edge1)
    // ==================================================================
    if (input.backgroundTrafficMbps > 0.0)
    {
        uint16_t bgPort = 9999;

        PacketSinkHelper bgSink("ns3::UdpSocketFactory",
                                 InetSocketAddress(Ipv4Address::GetAny(), bgPort));
        ApplicationContainer bgSinkApp = bgSink.Install(edgeServers.Get(0));
        bgSinkApp.Start(Seconds(0.0));
        bgSinkApp.Stop(Seconds(input.simulationTime));

        InetSocketAddress bgRemote(edgeAddrs[0], bgPort);
        bgRemote.SetTos(0x00);

        OnOffHelper bgOnOff("ns3::UdpSocketFactory", bgRemote);
        bgOnOff.SetConstantRate(
            DataRate(static_cast<uint64_t>(input.backgroundTrafficMbps * 1e6)),
            packetSize);
        bgOnOff.SetAttribute("OnTime",
            StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        bgOnOff.SetAttribute("OffTime",
            StringValue("ns3::ConstantRandomVariable[Constant=0]"));

        ApplicationContainer bgApp = bgOnOff.Install(cloudServer.Get(0));
        bgApp.Start(Seconds(1.0));
        bgApp.Stop(Seconds(input.simulationTime));
    }

    // ==================================================================
    //  10. Flow monitor
    // ==================================================================
    FlowMonitorHelper fmHelper;
    Ptr<FlowMonitor> flowMonitor = fmHelper.InstallAll();

    // ==================================================================
    //  11. Run
    // ==================================================================
    Simulator::Stop(Seconds(input.simulationTime + 1.0));
    Simulator::Run();

    // ==================================================================
    //  12. Collect metrics and write output
    // ==================================================================
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(fmHelper.GetClassifier());

    std::map<FlowId, FlowMonitor::FlowStats> flowStats = flowMonitor->GetFlowStats();

    std::vector<double> camProcessingDelayMs(TOTAL_CAMERAS, 0.0);
    for (uint32_t z = 0; z < NUM_ZONES; ++z)
    {
        double procDelay = ModelTypeToDelayMs(input.zones[z].modelType);
        for (uint32_t c = 0; c < CAMERAS_PER_ZONE; ++c)
        {
            camProcessingDelayMs[z * CAMERAS_PER_ZONE + c] = procDelay;
        }
    }

    json jOut;
    jOut["scenario_id"] = input.scenarioId;
    jOut["simulation_time"] = input.simulationTime;
    jOut["background_traffic_mbps"] = input.backgroundTrafficMbps;
    jOut["flows"] = json::array();

    for (auto& kv : flowStats)
    {
        Ipv4FlowClassifier::FiveTuple ft = classifier->FindFlow(kv.first);
        const FlowMonitor::FlowStats& fs = kv.second;

        uint16_t dstPort = ft.destinationPort;

        if (dstPort < SINK_PORT_BASE ||
            dstPort >= SINK_PORT_BASE + TOTAL_CAMERAS)
        {
            continue;
        }

        uint32_t camIdx    = dstPort - SINK_PORT_BASE;
        uint32_t zoneId    = camIdx / CAMERAS_PER_ZONE;
        uint32_t camInZone = camIdx % CAMERAS_PER_ZONE;

        if (zoneId >= NUM_ZONES) continue;

        const ZoneConfig& zc = input.zones[zoneId];

        double throughputMbps = 0.0;
        double avgDelayMs     = 0.0;
        double jitterMs       = 0.0;
        double lossRate       = 0.0;

        if (fs.rxBytes > 0)
        {
            double dur = (fs.timeLastRxPacket - fs.timeFirstTxPacket).GetSeconds();
            if (dur > 0.0)
                throughputMbps = (fs.rxBytes * 8.0) / (dur * 1e6);
        }

        if (fs.rxPackets > 1)
        {
            avgDelayMs = fs.delaySum.GetMilliSeconds()
                         / static_cast<double>(fs.rxPackets);
            jitterMs   = fs.jitterSum.GetMilliSeconds()
                         / static_cast<double>(fs.rxPackets - 1);
        }
        else if (fs.rxPackets == 1)
        {
            avgDelayMs = fs.delaySum.GetMilliSeconds();
            jitterMs   = 0.0;
        }

        if (fs.txPackets > 0)
        {
            lossRate = static_cast<double>(fs.lostPackets)
                       / static_cast<double>(fs.txPackets);
        }

        // Only add processing delay if packets were actually received
        if (fs.rxPackets > 0)
        {
            avgDelayMs += camProcessingDelayMs[camIdx];
        }

        std::ostringstream flowId;
        flowId << "cam_" << camIdx;

        json flowEntry;
        flowEntry["flow_id"]                 = flowId.str();
        flowEntry["cam_index"]               = camIdx;
        flowEntry["zone_id"]                 = zoneId;
        flowEntry["cam_in_zone"]             = camInZone;
        flowEntry["priority_class"]          = zc.priorityClass;
        flowEntry["model_type"]              = zc.modelType;
        flowEntry["processing_location"]     = zc.processingLocation;
        flowEntry["bitrate_configured_mbps"] = zc.bitrateMbps;
        flowEntry["throughput_mbps"]         = std::round(throughputMbps * 10000.0) / 10000.0;
        flowEntry["packet_loss_rate"]        = std::round(lossRate * 10000.0) / 10000.0;
        flowEntry["avg_delay_ms"]            = std::round(avgDelayMs * 10000.0) / 10000.0;
        flowEntry["jitter_ms"]               = std::round(jitterMs * 10000.0) / 10000.0;
        flowEntry["tx_packets"]              = fs.txPackets;
        flowEntry["rx_packets"]              = fs.rxPackets;
        flowEntry["lost_packets"]            = fs.lostPackets;

        jOut["flows"].push_back(flowEntry);
    }

    std::sort(jOut["flows"].begin(), jOut["flows"].end(),
              [](const json& a, const json& b)
              { return a["cam_index"].get<uint32_t>() < b["cam_index"].get<uint32_t>(); });

    jOut["total_flows"] = jOut["flows"].size();

    std::ofstream ofs(outputFile);
    if (!ofs.is_open())
    {
        std::cerr << "[ERROR] Cannot write to: " << outputFile << std::endl;
        Simulator::Destroy();
        return 1;
    }

    ofs << jOut.dump(2) << std::endl;
    ofs.close();

    std::cout << "[OK] " << jOut["total_flows"] << " flows written to: "
              << outputFile << std::endl;

    Simulator::Destroy();
    return 0;
}
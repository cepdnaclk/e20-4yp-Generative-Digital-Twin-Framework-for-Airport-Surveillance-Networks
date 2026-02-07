/*
 * Airport Surveillance Network Simulation (Fixed & Compile-Ready)
 * ------------------------------------------------------------------
 * Fixes:
 * 1. Replaced IsRunning() with IsPending()
 * 2. Moved QoS installation to CreateTopology (NodeContainer -> NetDeviceContainer fix)
 * 3. Fixed Constructor Initialization Order
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/error-model.h"
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>

using namespace ns3;
using json = nlohmann::json;

NS_LOG_COMPONENT_DEFINE("AirportSurveillance");

// ===========================================================================
// 1. CUSTOM VIDEO APPLICATION
// ===========================================================================

class VideoStreamApplication : public Application
{
public:
    VideoStreamApplication();
    virtual ~VideoStreamApplication();
    void Setup(Address primary, Address backup, uint32_t packetSize, DataRate dataRate, uint32_t priority);
    void SwitchToBackup();

private:
    virtual void StartApplication(void);
    virtual void StopApplication(void);
    void SendPacket(void);

    Ptr<Socket> m_socket;
    Address m_peer;        
    Address m_backupPeer;  
    bool m_usingBackup;    
    
    uint32_t m_packetSize;
    DataRate m_dataRate;
    EventId m_sendEvent;
    bool m_running;
    uint32_t m_priority;
};

// FIX 1: Correct Initialization Order (Matches Header Declaration)
VideoStreamApplication::VideoStreamApplication()
    : m_socket(0), 
      m_usingBackup(false), 
      m_packetSize(0), 
      m_dataRate(0), 
      m_running(false), 
      m_priority(0) 
{
}

VideoStreamApplication::~VideoStreamApplication() { m_socket = 0; }

void VideoStreamApplication::Setup(Address primary, Address backup, uint32_t packetSize, DataRate dataRate, uint32_t priority)
{
    m_peer = primary;
    m_backupPeer = backup;
    m_packetSize = packetSize;
    m_dataRate = dataRate;
    m_priority = priority;
    m_usingBackup = false;
}

void VideoStreamApplication::SwitchToBackup()
{
    if (!m_usingBackup && m_running) {
        m_peer = m_backupPeer; 
        m_usingBackup = true;
    }
}

void VideoStreamApplication::StartApplication(void)
{
    m_running = true;
    m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
    uint8_t tos = (m_priority == 1) ? 0xB8 : (m_priority == 2) ? 0x80 : 0x00;
    m_socket->SetIpTos(tos);
    SendPacket();
}

void VideoStreamApplication::StopApplication(void)
{
    m_running = false;
    // FIX 2: Replaced IsRunning() with IsPending()
    if (m_sendEvent.IsPending()) Simulator::Cancel(m_sendEvent);
    if (m_socket) m_socket->Close();
}

void VideoStreamApplication::SendPacket(void)
{
    Ptr<Packet> packet = Create<Packet>(m_packetSize);
    m_socket->SendTo(packet, 0, m_peer);
    
    if (m_running) {
        Time tNext(Seconds(m_packetSize * 8 / static_cast<double>(m_dataRate.GetBitRate())));
        m_sendEvent = Simulator::Schedule(tNext, &VideoStreamApplication::SendPacket, this);
    }
}

// ===========================================================================
// 2. MAIN SIMULATION
// ===========================================================================

class AirportSurveillanceSimulation
{
public:
    AirportSurveillanceSimulation(const std::string& scenarioFile);
    void Run(const std::string& outputFile);

private:
    void LoadScenario(const std::string& filename);
    void CreateTopology();
    void SetupApplications();
    void SetupBackgroundTraffic(); 
    // void SetupQoS(); // REMOVED: Moved inside CreateTopology
    void ExportResults(const std::string& outputFile);

    json m_scenario;
    NodeContainer m_nodes; 
    
    std::map<std::string, Ptr<Node>> m_nodeMap;
    std::map<Ipv4Address, std::string> m_ipToNameMap; 
    std::map<std::string, Ipv4Address> m_nameToIpMap;

    Ptr<FlowMonitor> m_flowMonitor;
    FlowMonitorHelper m_flowHelper;
};

AirportSurveillanceSimulation::AirportSurveillanceSimulation(const std::string& scenarioFile)
{
    LoadScenario(scenarioFile);
}

void AirportSurveillanceSimulation::LoadScenario(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        NS_FATAL_ERROR("Cannot open scenario file: " << filename);
    }
    file >> m_scenario;
}

void AirportSurveillanceSimulation::CreateTopology()
{
    auto GetOrCreateNode = [&](std::string id) -> Ptr<Node> {
        if (m_nodeMap.find(id) == m_nodeMap.end()) {
            Ptr<Node> node = CreateObject<Node>();
            m_nodes.Add(node);
            m_nodeMap[id] = node;
            InternetStackHelper internet;
            internet.Install(node);
        }
        return m_nodeMap[id];
    };

    // 1. Create Nodes
    for (const auto& cam : m_scenario["cameras"]) GetOrCreateNode(cam["id"]);
    for (const auto& edge : m_scenario["edge_servers"]) GetOrCreateNode(edge["id"]);
    for (const auto& cloud : m_scenario["cloud_endpoints"]) GetOrCreateNode(cloud["id"]);

    // 2. Create Links
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.255.0");

    // Helper for Traffic Control
    TrafficControlHelper tch;
    tch.SetRootQueueDisc("ns3::PrioQueueDisc");

    for (const auto& link : m_scenario["network_links"])
    {
        std::string src = link["src"];
        std::string dst = link["dst"];
        double bw = link["capacity_mbps"];
        double delay = link["latency_ms"];
        double loss = link["packet_loss_rate"]; 

        PointToPointHelper p2p;
        p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate(std::to_string((int)bw) + "Mbps")));
        p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(delay)));

        NetDeviceContainer devices = p2p.Install(m_nodeMap[src], m_nodeMap[dst]);

        // FIX 3: Install Traffic Control HERE on the devices
        tch.Install(devices);

        if (loss > 0.0) {
            Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
            em->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
            em->SetAttribute("ErrorRate", DoubleValue(loss));
            devices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em)); 
        }

        Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);
        
        m_ipToNameMap[interfaces.GetAddress(0)] = src;
        m_ipToNameMap[interfaces.GetAddress(1)] = dst;
        
        m_nameToIpMap[src] = interfaces.GetAddress(0);
        m_nameToIpMap[dst] = interfaces.GetAddress(1);

        ipv4.NewNetwork();
    }
    
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
}

void AirportSurveillanceSimulation::SetupBackgroundTraffic()
{
    uint16_t port = 5000;
    for (const auto& traffic : m_scenario["background_traffic"])
    {
        std::string src = traffic["src"];
        std::string dst = traffic["dst"];
        double rate = traffic["bitrate_mbps"];
        double start = traffic["start_time_s"];
        double duration = traffic["duration_s"];

        PacketSinkHelper sinkHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApp = sinkHelper.Install(m_nodeMap[dst]);
        sinkApp.Start(Seconds(0.0));
        sinkApp.Stop(Seconds(1000.0));

        Ipv4Address dstAddr = m_nameToIpMap[dst]; 
        OnOffHelper onoff("ns3::TcpSocketFactory", InetSocketAddress(dstAddr, port));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        onoff.SetAttribute("DataRate", DataRateValue(DataRate(std::to_string((int)rate) + "Mbps")));

        ApplicationContainer app = onoff.Install(m_nodeMap[src]); 
        app.Start(Seconds(start));
        app.Stop(Seconds(start + duration));
        
        port++; 
    }
}

void AirportSurveillanceSimulation::SetupApplications()
{
    uint16_t port = 9000;
    
    // 1. Setup Sinks
    for (auto const& [name, node] : m_nodeMap) {
        PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer apps = sink.Install(node);
        apps.Start(Seconds(0.0));
        apps.Stop(Seconds(65.0));
    }

    // 2. Setup Sources
    for (const auto& flow : m_scenario["flows"])
    {
        std::string src = flow["source"];
        std::string dst = flow["destination"];
        std::string backup = flow["backup_destination"]; 
        
        double bitrate = flow["bitrate_mbps"];
        int priority = flow["priority"];

        Ptr<Node> srcNode = m_nodeMap[src];
        
        Ipv4Address dstAddr = m_nameToIpMap[dst];
        Ipv4Address backupAddr = m_nameToIpMap[backup];

        Ptr<VideoStreamApplication> app = CreateObject<VideoStreamApplication>();
        app->Setup(InetSocketAddress(dstAddr, port), 
                   InetSocketAddress(backupAddr, port), 
                   1400, 
                   DataRate(std::to_string((int)(bitrate * 1e6)) + "bps"), 
                   priority);
        
        srcNode->AddApplication(app); 
        app->SetStartTime(Seconds(1.0));
        app->SetStopTime(Seconds(60.0));

        // Failover Logic
        for (const auto& fail : m_scenario["failures"]) {
            if (fail["target"] == dst && fail["failure_type"] == "shutdown") {
                double failTime = fail["start_time_s"];
                if(failTime < 60.0) {
                    Simulator::Schedule(Seconds(failTime + 2.0), &VideoStreamApplication::SwitchToBackup, app);
                }
            }
        }
    }
}

void AirportSurveillanceSimulation::ExportResults(const std::string& outputFile)
{
    m_flowMonitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowHelper.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = m_flowMonitor->GetFlowStats();
    
    json results;
    results["scenario_id"] = m_scenario["scenario_id"];
    results["flows"] = json::array();

    for (auto const& [flowId, stat] : stats)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flowId);
        
        if (t.destinationPort == 9000) 
        {
            std::string srcName = m_ipToNameMap[t.sourceAddress];
            std::string dstName = m_ipToNameMap[t.destinationAddress];
            
            for (const auto& flow : m_scenario["flows"]) {
                if (flow["source"] == srcName && flow["destination"] == dstName) {
                    
                    json flowResult;
                    flowResult["flow_id"] = flow["id"];
                    flowResult["zone"] = flow["zone"];
                    flowResult["priority"] = flow["priority"];
                    
                    double duration = stat.timeLastRxPacket.GetSeconds() - stat.timeFirstTxPacket.GetSeconds();
                    if (duration <= 0) duration = 1.0;

                    flowResult["throughput_mbps"] = (stat.rxBytes * 8.0) / duration / 1e6;
                    flowResult["packet_loss_rate"] = (double)stat.lostPackets / (stat.txPackets + 0.00001);
                    flowResult["avg_delay_ms"] = (stat.rxPackets > 0) ? (stat.delaySum.GetMilliSeconds() / stat.rxPackets) : 0.0;
                    
                    double jitter_ms = 0.0;
                    if (stat.rxPackets > 1) {
                         jitter_ms = stat.jitterSum.GetMilliSeconds() / (double)(stat.rxPackets - 1);
                    }
                    flowResult["jitter_ms"] = jitter_ms;

                    results["flows"].push_back(flowResult);
                    break; 
                }
            }
        }
    }

    std::ofstream out(outputFile);
    out << results.dump(4) << std::endl;
    NS_LOG_INFO("Results exported to " << outputFile);
}

void AirportSurveillanceSimulation::Run(const std::string& outputFile)
{
    CreateTopology();
    SetupBackgroundTraffic();
    SetupApplications();
    // SetupQoS(); // REMOVED: Now done inside CreateTopology

    m_flowMonitor = m_flowHelper.InstallAll();

    NS_LOG_INFO("Running simulation...");
    Simulator::Stop(Seconds(65.0));
    Simulator::Run();
    
    ExportResults(outputFile);
    Simulator::Destroy();
}

int main(int argc, char *argv[])
{
    std::string scenarioFile = "scenario_0000.json";
    std::string outputFile = "output/results_0000.json";

    CommandLine cmd;
    cmd.AddValue("scenario", "Input JSON file path", scenarioFile);
    cmd.AddValue("output", "Output JSON file path", outputFile);
    cmd.Parse(argc, argv);

    AirportSurveillanceSimulation sim(scenarioFile);
    sim.Run(outputFile);
    return 0;
}
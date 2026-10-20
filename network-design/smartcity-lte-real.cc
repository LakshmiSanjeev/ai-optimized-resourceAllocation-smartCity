#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/network-module.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/flow-monitor-module.h"

#include <fstream>
#include <map>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("SmartCityLTE");

int mainlte(int argc, char *argv[])
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    std::string schedulerType = "PF";
    uint16_t numberOfUes = 50;
    bool enableTraces = false;

    CommandLine cmd;
    cmd.AddValue("scheduler", "Scheduler type: PF / RR / MT", schedulerType);
    cmd.AddValue("nUes", "Number of UEs", numberOfUes);
    cmd.AddValue("enableTraces", "Enable LTE traces (true/false)", enableTraces);
    cmd.Parse(argc, argv);

    std::string scheduler;

    if (schedulerType == "PF")
     scheduler = "ns3::PfFfMacScheduler";
    else if (schedulerType == "RR")
     scheduler = "ns3::RrFfMacScheduler";
    else if (schedulerType == "MT")
     scheduler = "ns3::FdMtFfMacScheduler";
    else
    {
        NS_LOG_UNCOND("Invalid scheduler. Using PF.");
        scheduler = "ns3::PfFfMacScheduler";
    }

    uint32_t group1 = numberOfUes / 3;
    uint32_t group2 = 2 * numberOfUes / 3;
    double simTime = 20.0;

    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);

    lteHelper->SetSchedulerType(scheduler);

    Config::SetDefault("ns3::LteUePhy::NoiseFigure", DoubleValue(9.0));
    Config::SetDefault("ns3::LteEnbPhy::NoiseFigure", DoubleValue(5.0));

    lteHelper->SetAttribute("PathlossModel",
        StringValue("ns3::LogDistancePropagationLossModel"));

    Config::SetDefault("ns3::LogDistancePropagationLossModel::Exponent",
        DoubleValue(3.5));

    Config::SetDefault("ns3::LogDistancePropagationLossModel::ReferenceLoss",
        DoubleValue(30.0));

    lteHelper->SetAttribute("FadingModel",
        StringValue("ns3::TraceFadingLossModel"));

    lteHelper->SetFadingModelAttribute("TraceFilename",
        StringValue("src/lte/model/fading-traces/fading_trace_EPA_3kmph.fad"));

    lteHelper->SetFadingModelAttribute("TraceLength", TimeValue(Seconds(10.0)));

    lteHelper->SetFadingModelAttribute("SamplesNum", UintegerValue(10000));

    lteHelper->SetEnbDeviceAttribute("DlEarfcn", UintegerValue(100));
    lteHelper->SetEnbDeviceAttribute("UlEarfcn", UintegerValue(18100));
    lteHelper->SetEnbDeviceAttribute("DlBandwidth", UintegerValue(50));
    lteHelper->SetEnbDeviceAttribute("UlBandwidth", UintegerValue(50));

    NodeContainer enbNodes;
    enbNodes.Create(2);
    NodeContainer ueNodes;
    ueNodes.Create(numberOfUes);
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);

    MobilityHelper mobility;

    Ptr<ListPositionAllocator> enbPos = CreateObject<ListPositionAllocator>();
    enbPos->Add(Vector(250,250,0));
    enbPos->Add(Vector(750,250,0));

    mobility.SetPositionAllocator(enbPos);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(enbNodes);

    Ptr<RandomRectanglePositionAllocator> uePos = CreateObject<RandomRectanglePositionAllocator>();

    uePos->SetAttribute("X", StringValue("ns3::UniformRandomVariable[Min=0|Max=1000]"));
    uePos->SetAttribute("Y", StringValue("ns3::UniformRandomVariable[Min=0|Max=500]"));

    mobility.SetMobilityModel(
        "ns3::RandomWaypointMobilityModel",
        "Speed", StringValue("ns3::UniformRandomVariable[Min=5|Max=20]"),
        "Pause", StringValue("ns3::ConstantRandomVariable[Constant=0]"),
        "PositionAllocator", PointerValue(uePos));

    mobility.Install(ueNodes);

    InternetStackHelper internet;
    internet.Install(ueNodes);
    internet.Install(remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", StringValue("100Gbps"));
    p2ph.SetChannelAttribute("Delay", StringValue("10ms"));

    NetDeviceContainer internetDevices =
        p2ph.Install(epcHelper->GetPgwNode(), remoteHost);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0","255.0.0.0");

    Ipv4InterfaceContainer internetIpIfaces =
        ipv4h.Assign(internetDevices);

    Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress(1);

    NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueLteDevs  = lteHelper->InstallUeDevice(ueNodes);

    Ipv4InterfaceContainer ueIpIface =
        epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLteDevs));

    std::map<Ipv4Address, uint32_t> ipToUeMap;

    for (uint32_t u = 0; u < numberOfUes; ++u)
    {
        ipToUeMap[ueIpIface.GetAddress(u)] = u;
    }

    lteHelper->AttachToClosestEnb(ueLteDevs, enbLteDevs);

    Ptr<EpcTft> tft = Create<EpcTft>();

    EpcTft::PacketFilter pf;
    pf.localPortStart = 0;
    pf.localPortEnd = 65535;
    tft->Add(pf);

    EpsBearer bearerIoT(EpsBearer::NGBR_IMS);
    EpsBearer bearerVideo(EpsBearer::GBR_CONV_VIDEO);
    EpsBearer bearerMid(EpsBearer::NGBR_VIDEO_TCP_DEFAULT);

    for (uint32_t u = 0; u < numberOfUes; ++u)
    {
        Ptr<NetDevice> ueDevice = ueLteDevs.Get(u);
        if (u < group1)
        {
            lteHelper->ActivateDedicatedEpsBearer(ueDevice, bearerIoT, tft);
        }
        else if (u < group2)
        {
            lteHelper->ActivateDedicatedEpsBearer(ueDevice, bearerVideo, tft);
        }
        else
        {
            lteHelper->ActivateDedicatedEpsBearer(ueDevice, bearerMid, tft);
        }
    }

    Ipv4StaticRoutingHelper ipv4RoutingHelper;

    for (uint32_t u = 0; u < numberOfUes; ++u)
    {
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNodes.Get(u)->GetObject<Ipv4>());

        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());

    remoteHostStaticRouting->AddNetworkRouteTo(
        Ipv4Address("7.0.0.0"),
        Ipv4Mask("255.0.0.0"),
        1);

    uint16_t dlPort = 1000;
    uint16_t ulPort = 2000;

    PacketSinkHelper ulSink("ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), ulPort));
    ulSink.Install(remoteHost).Start(Seconds(1.0));

    for(uint32_t u = 0; u < numberOfUes; ++u)
    {
        int interval = (u < group1) ? 100 : (u < group2 ? 10 : 20);
        int size     = (u < group1) ? 200 : (u < group2 ? 1024 : 512);

        UdpClientHelper dlClient(ueIpIface.GetAddress(u), dlPort);
        dlClient.SetAttribute("MaxPackets", UintegerValue(1000000));
        dlClient.SetAttribute("Interval", TimeValue(MilliSeconds(interval)));
        dlClient.SetAttribute("PacketSize", UintegerValue(size));
        dlClient.Install(remoteHost).Start(Seconds(1.0));

        PacketSinkHelper dlSink("ns3::UdpSocketFactory",
            InetSocketAddress(Ipv4Address::GetAny(), dlPort));
        dlSink.Install(ueNodes.Get(u)).Start(Seconds(1.0));

        UdpClientHelper ulClient(remoteHostAddr, ulPort);
        ulClient.SetAttribute("MaxPackets", UintegerValue(1000000));
        ulClient.SetAttribute("Interval", TimeValue(MilliSeconds(interval)));
        ulClient.SetAttribute("PacketSize", UintegerValue(size));
        ulClient.Install(ueNodes.Get(u)).Start(Seconds(1.0));
    }

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    if (enableTraces)
    {
        lteHelper->EnablePhyTraces();
        lteHelper->EnableMacTraces();
        lteHelper->EnableRlcTraces();
        lteHelper->EnablePdcpTraces();
    }

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    monitor->CheckForLostPackets();

    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());

    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    std::ofstream outFile;
    std::string filename = "FMResults_" + schedulerType + "_" + std::to_string(numberOfUes) + ".csv";
    outFile.open(filename);

    outFile << "FlowID,UE_ID,TrafficClass,SrcIP,DstIP,"
            << "TxPackets,RxPackets,LostPackets,"
            << "TxBytes,RxBytes,Throughput(Mbps),Delay(ms),Jitter(ms)\n";

    for (auto &flow : stats)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flow.first);

        int ueId = -1;

        if (ipToUeMap.find(t.sourceAddress) != ipToUeMap.end())
        {
            ueId = ipToUeMap[t.sourceAddress];
        }
        else if (ipToUeMap.find(t.destinationAddress) != ipToUeMap.end())
        {
            ueId = ipToUeMap[t.destinationAddress];
        }

        int trafficClass = -1;

        if (ueId != -1)
        {
            if (ueId < group1)
                trafficClass = 0;
            else if (ueId < group2)
                trafficClass = 1;
            else
                trafficClass = 2;
        }

        double throughput = 0;
        if (flow.second.timeLastRxPacket.GetSeconds() >
            flow.second.timeFirstTxPacket.GetSeconds())
        {
            throughput = flow.second.rxBytes * 8.0 /
                (flow.second.timeLastRxPacket.GetSeconds() -
                flow.second.timeFirstTxPacket.GetSeconds()) / 1e6;
        }

        double delay = (flow.second.rxPackets > 0) ?
            (flow.second.delaySum.GetSeconds() / flow.second.rxPackets) * 1000 : 0;

        double jitter = (flow.second.rxPackets > 0) ?
            (flow.second.jitterSum.GetSeconds() / flow.second.rxPackets) * 1000 : 0;

        outFile << flow.first << ","
                << ueId << ","
                << trafficClass << ","
                << t.sourceAddress << ","
                << t.destinationAddress << ","
                << flow.second.txPackets << ","
                << flow.second.rxPackets << ","
                << flow.second.lostPackets << ","
                << flow.second.txBytes << ","
                << flow.second.rxBytes << ","
                << throughput << ","
                << delay << ","
                << jitter << "\n";
    }

    outFile.close();

    Simulator::Destroy();
    return 0;
}
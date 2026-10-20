#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/ipv4-static-routing-helper.h"
#include <array>
#include <fstream>
#include <limits>
#include <map>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("SmartCity5G");

int main5g(int argc, char* argv[])
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    std::string schedulerType = "PF";
    uint16_t    nUes          = 30;
    bool        enableTraces  = false;

    CommandLine cmd(__FILE__);
    cmd.AddValue("scheduler",    "MAC scheduler: PF | RR | MT", schedulerType);
    cmd.AddValue("nUes",         "Number of UEs",               nUes);
    cmd.AddValue("enableTraces", "Enable NR PHY/MAC traces",    enableTraces);
    cmd.Parse(argc, argv);

    std::string schedulerTid;
    if      (schedulerType == "RR") schedulerTid = "ns3::NrMacSchedulerTdmaRR";
    else if (schedulerType == "MT") schedulerTid = "ns3::NrMacSchedulerTdmaMT";
    else                             schedulerTid = "ns3::NrMacSchedulerTdmaPF";

    const uint32_t N       = static_cast<uint32_t>(nUes);
    const uint32_t group1  = std::max(1u,           N / 3u);
    const uint32_t group2  = std::max(group1 + 1u, (2u * N) / 3u);
    const double   simTime = 15.0;
    const uint16_t dlPort  = 1000;
    const uint16_t ulPort  = 2000;

    const std::string tag = schedulerType + "_" + std::to_string(N) + "UE";

    Ptr<NrPointToPointEpcHelper> epcHelper =
        CreateObject<NrPointToPointEpcHelper>();

    Ptr<IdealBeamformingHelper> beamHelper =
        CreateObject<IdealBeamformingHelper>();
    beamHelper->SetAttribute("BeamformingMethod",
        TypeIdValue(TypeId::LookupByName("ns3::DirectPathBeamforming")));

    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetBeamformingHelper(beamHelper);
    nrHelper->SetEpcHelper(epcHelper);
    nrHelper->SetSchedulerTypeId(TypeId::LookupByName(schedulerTid));

    nrHelper->SetGnbAntennaAttribute("NumRows",    UintegerValue(1));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute ("NumRows",    UintegerValue(1));
    nrHelper->SetUeAntennaAttribute ("NumColumns", UintegerValue(1));

    nrHelper->SetGnbPhyAttribute("NoiseFigure", DoubleValue(5.0));
    nrHelper->SetUePhyAttribute ("NoiseFigure", DoubleValue(7.0));

    CcBwpCreator::SimpleOperationBandConf bandConf(2.0e9, 100e6, 3);

    auto [totalBw, allBwps] = nrHelper->CreateBandwidthParts(
        {bandConf},
        "UMa",
        "Default",
        "ThreeGpp"
    );
    (void)totalBw;

    NodeContainer gnbNodes, ueNodes, rhContainer;
    gnbNodes.Create(2);
    ueNodes.Create(N);
    rhContainer.Create(1);
    Ptr<Node> remoteHost = rhContainer.Get(0);

    MobilityHelper mob;

    {
        Ptr<ListPositionAllocator> pos = CreateObject<ListPositionAllocator>();
        pos->Add(Vector(250.0, 250.0, 30.0));
        pos->Add(Vector(750.0, 250.0, 30.0));
        mob.SetPositionAllocator(pos);
        mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        mob.Install(gnbNodes);
    }

    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.Install(rhContainer);
    mob.Install(epcHelper->GetPgwNode());

    {
        Ptr<RandomRectanglePositionAllocator> initAlloc =
            CreateObject<RandomRectanglePositionAllocator>();
        initAlloc->SetAttribute("X",
            StringValue("ns3::UniformRandomVariable[Min=0|Max=1000]"));
        initAlloc->SetAttribute("Y",
            StringValue("ns3::UniformRandomVariable[Min=0|Max=500]"));

        Ptr<RandomRectanglePositionAllocator> wpAlloc =
            CreateObject<RandomRectanglePositionAllocator>();
        wpAlloc->SetAttribute("X",
            StringValue("ns3::UniformRandomVariable[Min=0|Max=1000]"));
        wpAlloc->SetAttribute("Y",
            StringValue("ns3::UniformRandomVariable[Min=0|Max=500]"));

        mob.SetPositionAllocator(initAlloc);
        mob.SetMobilityModel(
            "ns3::RandomWaypointMobilityModel",
            "Speed",  StringValue("ns3::UniformRandomVariable[Min=5|Max=20]"),
            "Pause",  StringValue("ns3::ConstantRandomVariable[Constant=1.0]"),
            "PositionAllocator", PointerValue(wpAlloc));
        mob.Install(ueNodes);
    }

    InternetStackHelper internet;
    internet.Install(ueNodes);
    internet.Install(rhContainer);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute ("DataRate", StringValue("100Gbps"));
    p2p.SetChannelAttribute("Delay",    StringValue("10ms"));
    NetDeviceContainer pgwRhDevs = p2p.Install(epcHelper->GetPgwNode(), remoteHost);

    Ipv4AddressHelper ipHelper;
    ipHelper.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer pgwRhIfaces = ipHelper.Assign(pgwRhDevs);
    Ipv4Address remoteHostAddr = pgwRhIfaces.GetAddress(1);

    NetDeviceContainer gnbDevs = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueDevs  = nrHelper->InstallUeDevice (ueNodes,  allBwps);

    int64_t rngStream = 1;
    rngStream += nrHelper->AssignStreams(gnbDevs, rngStream);
    rngStream += nrHelper->AssignStreams(ueDevs,  rngStream);

    Ipv4InterfaceContainer ueIpIfaces =
        epcHelper->AssignUeIpv4Address(ueDevs);

    std::map<Ipv4Address, uint32_t> ip2ue;
    for (uint32_t u = 0; u < N; ++u)
        ip2ue[ueIpIfaces.GetAddress(u)] = u;

    for (uint32_t u = 0; u < N; ++u)
    {
        Ptr<MobilityModel> uMob = ueNodes.Get(u)->GetObject<MobilityModel>();
        double   minDist    = std::numeric_limits<double>::max();
        uint32_t bestGnb    = 0;
        for (uint32_t g = 0; g < gnbNodes.GetN(); ++g)
        {
            double d = uMob->GetDistanceFrom(
                gnbNodes.Get(g)->GetObject<MobilityModel>());
            if (d < minDist) { minDist = d; bestGnb = g; }
        }
        nrHelper->AttachToGnb(ueDevs.Get(u), gnbDevs.Get(bestGnb));
    }

    Ipv4StaticRoutingHelper staticRouting;

    for (uint32_t u = 0; u < N; ++u)
    {
        auto ueRoute = staticRouting.GetStaticRouting(
            ueNodes.Get(u)->GetObject<Ipv4>());
        ueRoute->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    auto rhRoute = staticRouting.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    rhRoute->AddNetworkRouteTo(
        Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);


    for (uint32_t u = 0; u < N; ++u)
    {
        Ptr<NrQosRule> rule = Create<NrQosRule>();
        NrQosRule::PacketFilter pf;
        pf.localPortStart = dlPort;
        pf.localPortEnd   = dlPort;
        rule->Add(pf);

        if (u < group1)
        {
            NrQosFlow flow(NrQosFlow::NGBR_VIDEO_TCP_DEFAULT);
            nrHelper->ActivateDedicatedQosFlow(ueDevs.Get(u), flow, rule);
        }
        else if (u < group2)
        {
            NrGbrQosInformation qos;
            qos.gbrDl = 20'000'000ULL;
            qos.gbrUl =  5'000'000ULL;
            qos.mbrDl = 30'000'000ULL;
            qos.mbrUl = 10'000'000ULL;
            NrQosFlow flow(NrQosFlow::GBR_NON_CONV_VIDEO, qos);
            nrHelper->ActivateDedicatedQosFlow(ueDevs.Get(u), flow, rule);
        }
        else
        {
            NrQosFlow flow(NrQosFlow::NGBR_VOICE_VIDEO_GAMING);
            nrHelper->ActivateDedicatedQosFlow(ueDevs.Get(u), flow, rule);
        }
    }

    PacketSinkHelper ulSinkH("ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), ulPort));
    ulSinkH.Install(remoteHost).Start(Seconds(1.0));

    for (uint32_t u = 0; u < N; ++u)
    {
        const int    interval = (u < group1) ? 100 : (u < group2 ? 10 : 20);
        const int    pktSize  = (u < group1) ? 200 : (u < group2 ? 1024 : 512);
        const double dlStart  = 2.0 + (u * 0.05);
        const double ulStart  = 2.5 + (u * 0.05);

        PacketSinkHelper dlSinkH("ns3::UdpSocketFactory",
            InetSocketAddress(Ipv4Address::GetAny(), dlPort));
        dlSinkH.Install(ueNodes.Get(u)).Start(Seconds(1.0));

        UdpClientHelper dlClient(ueIpIfaces.GetAddress(u), dlPort);
        dlClient.SetAttribute("MaxPackets", UintegerValue(1'000'000));
        dlClient.SetAttribute("Interval",   TimeValue(MilliSeconds(interval)));
        dlClient.SetAttribute("PacketSize", UintegerValue(pktSize));
        dlClient.Install(remoteHost).Start(Seconds(dlStart));

        UdpClientHelper ulClient(remoteHostAddr, ulPort);
        ulClient.SetAttribute("MaxPackets", UintegerValue(1'000'000));
        ulClient.SetAttribute("Interval",   TimeValue(MilliSeconds(interval)));
        ulClient.SetAttribute("PacketSize", UintegerValue(pktSize));
        ulClient.Install(ueNodes.Get(u)).Start(Seconds(ulStart));
    }

    FlowMonitorHelper fmHelper;
    Ptr<FlowMonitor>  monitor = fmHelper.InstallAll();

    if (enableTraces)
        nrHelper->EnableTraces();
    
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(fmHelper.GetClassifier());
    auto stats = monitor->GetFlowStats();

    auto resolveFlow =
        [&](const Ipv4FlowClassifier::FiveTuple& t,
            int& ueId, bool& isDL)
    {
        ueId = -1; isDL = false;
        auto itDst = ip2ue.find(t.destinationAddress);
        auto itSrc = ip2ue.find(t.sourceAddress);
        if (itDst != ip2ue.end()) { ueId = (int)itDst->second; isDL = true;  }
        else if (itSrc != ip2ue.end()) { ueId = (int)itSrc->second; isDL = false; }
    };

    auto getClass = [&](uint32_t uid, int& tc, int& qi)
    {
        if      (uid < group1) { tc = 0; qi = 9; }
        else if (uid < group2) { tc = 1; qi = 4; }
        else                   { tc = 2; qi = 7; }
    };

    auto calcMetrics =
        [](const FlowMonitor::FlowStats& s,
           double& tput, double& delay,
           double& jitter, double& lossPct)
    {
        double dur = s.timeLastRxPacket.GetSeconds()
                   - s.timeFirstTxPacket.GetSeconds();
        tput    = (dur > 0 && s.rxBytes > 0)
                    ? (s.rxBytes * 8.0 / dur / 1e6) : 0.0;
        delay   = s.rxPackets > 0
                    ? (s.delaySum.GetSeconds()  / s.rxPackets * 1e3) : 0.0;
        jitter  = s.rxPackets > 0
                    ? (s.jitterSum.GetSeconds() / s.rxPackets * 1e3) : 0.0;
        lossPct = s.txPackets > 0
                    ? (100.0 * s.lostPackets / s.txPackets) : 0.0;
    };

    {
        std::ofstream f("FlowDetails_5G_" + tag + ".csv");
        f << "FlowID,Direction,UE_ID,TrafficClass,FiveQI,"
             "SrcIP,DstIP,SrcPort,DstPort,"
             "TxPackets,RxPackets,LostPackets,LossPct(%),"
             "TxBytes,RxBytes,Throughput(Mbps),AvgDelay(ms),AvgJitter(ms)\n";

        for (auto& [fid, fs] : stats)
        {
            auto t = classifier->FindFlow(fid);
            int  ueId; bool isDL;
            resolveFlow(t, ueId, isDL);

            int tc = -1, qi = -1;
            if (ueId >= 0) getClass((uint32_t)ueId, tc, qi);

            double tput, delay, jitter, lossPct;
            calcMetrics(fs, tput, delay, jitter, lossPct);

            f << fid << ","
              << (ueId < 0 ? "OTHER" : (isDL ? "DL" : "UL")) << ","
              << ueId << "," << tc << "," << qi << ","
              << t.sourceAddress << "," << t.destinationAddress << ","
              << t.sourcePort    << "," << t.destinationPort    << ","
              << fs.txPackets    << "," << fs.rxPackets         << ","
              << fs.lostPackets  << "," << lossPct              << ","
              << fs.txBytes      << "," << fs.rxBytes           << ","
              << tput            << "," << delay                << ","
              << jitter          << "\n";
        }
    }

    struct UeRow {
        double dlTput=0, ulTput=0;
        double dlDelay=0, ulDelay=0;
        double dlJitter=0, ulJitter=0;
        double dlLoss=0, ulLoss=0;
    };
    std::map<int, UeRow> ueRows;

    for (auto& [fid, fs] : stats)
    {
        auto t = classifier->FindFlow(fid);
        int  ueId; bool isDL;
        resolveFlow(t, ueId, isDL);
        if (ueId < 0) continue;

        double tput, delay, jitter, lossPct;
        calcMetrics(fs, tput, delay, jitter, lossPct);

        UeRow& r = ueRows[ueId];
        if (isDL) { r.dlTput=tput; r.dlDelay=delay; r.dlJitter=jitter; r.dlLoss=lossPct; }
        else      { r.ulTput=tput; r.ulDelay=delay; r.ulJitter=jitter; r.ulLoss=lossPct; }
    }

    {
        std::ofstream f("UE_Summary_5G_" + tag + ".csv");
        f << "UE_ID,TrafficClass,FiveQI,"
             "DL_Tput(Mbps),UL_Tput(Mbps),"
             "DL_Delay(ms),UL_Delay(ms),"
             "DL_Jitter(ms),UL_Jitter(ms),"
             "DL_Loss(%),UL_Loss(%)\n";

        for (uint32_t u = 0; u < N; ++u)
        {
            int tc, qi; getClass(u, tc, qi);
            UeRow& r = ueRows[(int)u];
            f << u  << "," << tc << "," << qi << ","
              << r.dlTput   << "," << r.ulTput   << ","
              << r.dlDelay  << "," << r.ulDelay  << ","
              << r.dlJitter << "," << r.ulJitter << ","
              << r.dlLoss   << "," << r.ulLoss   << "\n";
        }
    }

    struct ClassAgg {
        uint32_t count = 0;
        double sumDLTput=0, sumULTput=0;
        double sumDLDelay=0, sumULDelay=0;
        double sumDLJitter=0, sumULJitter=0;
        double sumDLLoss=0,  sumULLoss=0;
    };
    std::array<ClassAgg, 3> ca{};

    for (uint32_t u = 0; u < N; ++u)
    {
        int tc, qi; getClass(u, tc, qi);
        UeRow& r = ueRows[(int)u];
        auto&  a = ca[(size_t)tc];
        ++a.count;
        a.sumDLTput   += r.dlTput;   a.sumULTput   += r.ulTput;
        a.sumDLDelay  += r.dlDelay;  a.sumULDelay  += r.ulDelay;
        a.sumDLJitter += r.dlJitter; a.sumULJitter += r.ulJitter;
        a.sumDLLoss   += r.dlLoss;   a.sumULLoss   += r.ulLoss;
    }

    {
        std::ofstream f("ClassSummary_5G_" + tag + ".csv");
        f << "TrafficClass,Description,FiveQI,NumUEs,"
             "Avg_DL_Tput(Mbps),Avg_UL_Tput(Mbps),"
             "Avg_DL_Delay(ms),Avg_UL_Delay(ms),"
             "Avg_DL_Jitter(ms),Avg_UL_Jitter(ms),"
             "Avg_DL_Loss(%),Avg_UL_Loss(%)\n";

        static const char* classNames[] =
            { "IoT_Sensors", "Video_Streaming", "Mixed_Data" };
        static const int   classQIs[]   = { 9, 4, 7 };

        for (int tc = 0; tc < 3; ++tc)
        {
            const auto& a = ca[(size_t)tc];
            const uint32_t k = std::max(1u, a.count);
            f << tc                    << ","
              << classNames[tc]        << ","
              << classQIs[tc]          << ","
              << a.count               << ","
              << a.sumDLTput   / k     << ","
              << a.sumULTput   / k     << ","
              << a.sumDLDelay  / k     << ","
              << a.sumULDelay  / k     << ","
              << a.sumDLJitter / k     << ","
              << a.sumULJitter / k     << ","
              << a.sumDLLoss   / k     << ","
              << a.sumULLoss   / k     << "\n";
        }
    }

    Simulator::Destroy();
    return 0;
}
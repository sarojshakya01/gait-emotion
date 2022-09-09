from cProfile import label
import matplotlib.pyplot as plt

# vloss_100 = [1.3594646128741177, 1.3028842535885898, 1.2811711051247336, 1.3137716488404707, 1.2057728442278774, 1.2716325196352871, 1.2490556185895747, 1.2326924800872803, 1.2196717749942432, 1.20520788973028, 1.1088405630805276, 1.0931811170144514, 1.1189604130658237, 1.098419189453125, 1.080079891464927, 1.117203262719241, 1.1272143179720098, 1.100495782765475, 1.0980411984703757, 1.1034903743050315]
# tloss_100 = [1.382563822340257, 1.2949387720315764, 1.2642565629269817, 1.2679657564304843, 1.260542616985812, 1.260927105304038, 1.2443646523031857, 1.2411560888337616, 1.2074569770605257, 1.210776112457313, 1.1457594969485065, 1.1356091086227116, 1.1160195157079413, 1.1153916757885773, 1.130370859462436, 1.114001480659636, 1.1203800186072246, 1.1150745725867772, 1.108703617412265, 1.1046358063669488]

# epoch_100 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

vloss_300 = [
    1.3708703084425493, 1.2667102271860295, 1.284005652774464, 1.241467616774819, 1.1642359711907126, 1.345854396169836, 1.2274390242316506, 1.1712676178325305, 1.1354600191116333, 1.3072869723493403, 1.1950760158625515, 1.302446354519237, 1.147543495351618, 1.3325752995230935, 1.2079159454865889, 1.1190025156194514, 1.129153007810766, 1.2832673029466108, 1.195877123962749, 1.2180692878636448, 1.214005242694508, 1.2126541625369678, 1.207244797186418, 1.1300914450125261, 1.108481076630679,
    1.167522506280379, 1.0915803800929675, 1.0798286199569702, 1.2386675802144138, 1.1207398934797808, 1.0902009497989307, 1.0674009323120117, 1.0721372203393416, 1.0607945106246255, 1.0548807165839456, 1.0917310335419395, 1.0489581173116511, 1.0567652149633928, 1.0457590330730786, 1.0574071569876238, 1.0494158864021301, 1.0357561653310603, 1.0534902702678333, 1.042932933027094, 1.0649949745698408, 1.0605808171358975, 1.06214209578254, 1.053923953663219, 1.062775113365867, 1.0529394800012761,
    1.053109423680739, 1.0488523786718196, 1.0711899508129468, 1.0488861094821582, 1.04162827946923, 1.0671174092726274, 1.0480738444761797, 1.0520674857226284, 1.040572096001018, 1.0596689961173318
]

tloss_300 = [
    1.3715067759598836, 1.2704463618816715, 1.2251603744997837, 1.2042471371074714, 1.201395872205791, 1.170063352230752, 1.1847970503391605, 1.1640012169828509, 1.162044089029331, 1.1413900982035268, 1.1784758986813006, 1.1669567298180987, 1.1598324114733403, 1.1450397401753039, 1.168579636824013, 1.1412798384628673, 1.137671123636831, 1.1608055900819232, 1.1354148653474185, 1.12276028996647, 1.167265267655401, 1.1333747175660465, 1.1391268927272002, 1.1179327498568166, 1.1525163809851844,
    1.13492619814259, 1.1124995669516007, 1.1103394302991356, 1.1193809013555545, 1.105410078964611, 1.0833723751625213, 1.0887024137053158, 1.065708275478665, 1.0880401411859115, 1.0784343876460991, 1.0734899191573115, 1.0705740327882294, 1.0780565886214228, 1.06408695773323, 1.064465056551565, 1.0777677238577663, 1.054654067105586, 1.057975672259189, 1.0510960646194987, 1.0631225834978688, 1.0507859339808474, 1.05610856384334, 1.049915631218712, 1.066440216385492, 1.0579779856275804,
    1.0749271476622855, 1.0513697626567122, 1.0593559848199976, 1.054486104757479, 1.0568288647302306, 1.0459402722887474, 1.0510607521132667, 1.0553080764147316, 1.062429631110465, 1.0410178958779515
]
epoch_300 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 205, 200, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300]

vloss_500 = [
    1.4412228194150059, 1.3800832358273594, 1.4223091819069602, 1.2363638227636164, 1.2709077108990063, 1.2195626009594311, 1.4189311916177922, 1.4374294497750022, 1.158415751023726, 1.2453544519164346, 1.282350399277427, 1.2290555943142285, 1.2958325364372947, 1.1685229756615378, 1.2147087183865635, 1.1678244038061663, 1.272662948478352, 1.2258352745663037, 1.2561736865477129, 1.186538273637945, 1.19978928565979, 1.1483158523386174, 1.1864346319978887, 1.2736443823034114, 1.1570171876387163,
    1.2993215376680547, 1.298413872718811, 1.2429269768974998, 1.1969900347969749, 1.30232205174186, 1.1926071535457263, 1.2269099192185835, 1.1715461286631497, 1.1995186480608853, 1.2383234663443132, 1.1398843255910007, 1.136084253137762, 1.2322062362324109, 1.206367167559537, 1.196933236989108, 1.1366428624499927, 1.1650970578193665, 1.1673324704170227, 1.1419714472510598, 1.2476815472949634, 1.0906731648878618, 1.1071855696764858, 1.157077426260168, 1.1336948167194019, 1.0845149592919783,
    1.1015182516791604, 1.1005419817837803, 1.1051128723404624, 1.1002791253003208, 1.0867998058145696, 1.0764311281117527, 1.0945826985619285, 1.1031499125740745, 1.0716268149289219, 1.0941895734180103, 1.0702724998647517, 1.0965268449349836, 1.0859449885108254, 1.0881925442002036, 1.1031197038563816, 1.0907427885315635, 1.1047794493761929, 1.1018757820129395, 1.1164204424077815, 1.0828157284043052, 1.1127164526419207, 1.0738192038102583, 1.1085725155743686, 1.0679215843027288,
    1.0956897302107378, 1.0882698134942488, 1.097997177730907, 1.1047495982863687, 1.1413022550669583, 1.101649891246449, 1.086478422988545, 1.0789633664217861, 1.0820315046743914, 1.088017761707306, 1.108345698226582, 1.080896881493655, 1.0855410207401623, 1.1007427464831958, 1.1476479877125134, 1.1021988662806423, 1.0910151546651667, 1.0784552639180964, 1.1105304902250117, 1.1016581058502197, 1.0939998518336902, 1.0894356640902432, 1.1038546508008784, 1.0984348925677212, 1.0860698927532544,
    1.0925199768759988
]

tloss_500 = [
    1.3533977128491543, 1.2637320420529583, 1.2118146667386045, 1.2101048490788677, 1.2063546611530946, 1.202251141614253, 1.1864813518996287, 1.1791216713367123, 1.1622779882780396, 1.1609210006081232, 1.1603995071779383, 1.1550082818116292, 1.1727338598506285, 1.1447615399219022, 1.1500160711826664, 1.151041842923306, 1.151495079592903, 1.1534267870506438, 1.1213076875941588, 1.1556541241041505, 1.1411761942476328, 1.1370748570649931, 1.1386025518474012, 1.1592045467678864, 1.1322020811609703,
    1.1299103150273313, 1.1132583671277112, 1.1483094745343274, 1.125430415172388, 1.129202161685075, 1.1355053469686225, 1.1259384633290885, 1.1380664576398265, 1.11159082568518, 1.1075699087416773, 1.1117974578744114, 1.1058754873747874, 1.1453947076703062, 1.1153055081273069, 1.1011704041226076, 1.107150148047079, 1.0916760711386653, 1.107702605795152, 1.1113774044678943, 1.0905238260136974, 1.1119864240731343, 1.0929455603703413, 1.0585065006029488, 1.1038207399963151, 1.0905354672139234,
    1.0981570848143927, 1.0569922221769201, 1.0719376920473458, 1.0539162690096562, 1.0423847253959957, 1.0581384153649358, 1.0449838402247664, 1.0384057901873447, 1.0451783508357435, 1.057367498331731, 1.0322070888953634, 1.0574736477124809, 1.0359493257975814, 1.049544090091592, 1.0481789247824413, 1.0385716203415747, 1.0337212646361624, 1.0409873499728666, 1.0523551484145741, 1.0260743658141334, 1.0468287485660892, 1.0243435673194357, 1.0415404928792822, 1.05685557351254, 1.0265385827215592,
    1.0295512410673764, 1.0232833417335359, 1.023727314897103, 1.0367342776591235, 1.0195637097453127, 1.0245953863210018, 1.0185882602587786, 1.0349448045881668, 1.0208789588201164, 1.032528822374816, 1.0322316135510359, 1.0112182061270911, 1.0255562663078308, 1.0262447666413714, 1.034012797445354, 1.0151536057491113, 1.011873202748818, 1.0294209819028872, 1.030220626014294, 1.0175348903873178, 1.0223826925353248, 1.0347036801942504, 1.0159865293172325, 1.0182098663679444, 1.013671329706022
]

epoch_500 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500]

vloss_all_500 = [
    1.1504792352517446, 1.0921313484509787, 1.1790277143319448, 1.003836743036906, 1.0225238998730977, 1.0078235248724619, 1.0800772647062937, 1.0363804340362548, 1.0087853054205576, 1.0011878768603006, 1.0076103230317435, 1.0139918009440103, 0.9911994437376658, 1.0398301561673482, 1.0467646737893423, 1.0312431752681732, 1.0775954027970631, 1.016196874777476, 1.0002654333909353, 0.9846303542455037, 0.9646901309490203, 1.1510281145572663, 0.9822270055611928, 1.001184054215749, 0.9811248064041138,
    0.982811443010966, 0.9912758111953736, 0.9720382630825043, 0.9616580069065094, 0.9941408336162567, 0.9912265320618947, 1.0243260582288107, 1.0475447535514832, 0.9760077675183614, 0.9688852270444234, 1.0121095021565756, 0.9855707347393036, 1.012030259768168, 0.9792547086874644, 1.0240184863408406, 0.9598931849002839, 0.9612778385480245, 0.9652143975098928, 0.9852770706017812, 0.9770042101542155, 0.9891951700051625, 0.9548976858456929, 0.9457166155179342, 0.9763259629408518,
    0.9710513254006704, 0.9528450071811676, 0.9431255201498667, 0.9604142169157664, 0.9552798807621002, 0.9517510771751404, 0.9526715437571208, 0.9567676544189453, 0.9541936039924621, 0.9461936314900716, 0.9482045710086823, 0.9446568111578624, 0.9460059305032095, 0.9394320706526439, 0.9398513734340668, 0.9503706653912862, 0.9701319932937622, 0.9505238215128581, 0.9520043174425761, 0.9468936145305633, 0.9347205102443695, 0.9479523420333862, 0.9411634008089701, 0.9382869323094686,
    0.91651345094045, 0.9520218849182129, 0.9133778015772501, 0.9194203754266103, 0.9226006964842478, 0.9270990232626597, 0.9166491210460663, 0.9267237683137258, 0.9175080597400666, 0.9166453897953033, 0.9130071679751078, 0.9161192576090494, 0.9088765343030294, 0.921176822980245, 0.9200195928414663, 0.9199720919132233, 0.9163562536239624, 0.9201851665973664, 0.907879517475764, 0.9178532063961029, 0.9258634865283966, 0.922865508000056, 0.9156379183133443, 0.9190870026747385, 0.9233672300974528,
    0.9083486258983612, 0.9206456502278646
]
tloss_all_500 = [
    1.1780562602389943, 1.1241512597690928, 1.0939102268218994, 1.067500315579501, 1.0669276007738981, 1.0536180914532054, 1.0558383720571345, 1.0534374447302384, 1.053003588589755, 1.0370955499735746, 1.0323100313273343, 1.0357449735294688, 1.0279988934777, 1.0183250882408836, 1.0343159359151666, 1.0370697788758712, 1.0339353747801348, 1.0192865616625006, 1.0322376539490439, 1.0226371812820434, 1.018159481828863, 1.0262608781727878, 1.0151695290478793, 1.024337280880321, 1.0158321048996666,
    1.0236826809969815, 1.0147851881113918, 1.0335941022092645, 1.0116594290733338, 1.0110148451545022, 1.026826903820038, 1.018426910313693, 1.0308648620952259, 1.027508403821425, 1.0185455359112132, 1.0158094724741848, 1.0200343119014392, 1.0250037266991354, 1.0167236956683072, 1.0130390305952592, 1.0149255280061202, 1.0202081628279251, 1.0184974093870682, 1.0226309004696932, 1.0135267099467191, 1.0177486558393998, 1.0097504810853437, 1.01104867848483, 1.0148317131129179, 1.0084129144928673,
    0.9949466148289767, 0.990391181382266, 0.9855902357534929, 0.9877573791417209, 0.9818459588831121, 0.9841724759882147, 0.978047338182276, 0.9785092137076637, 0.9817332319779829, 0.9717277370799672, 0.9685203920711171, 0.974979905648665, 0.9713510931621898, 0.9720216150717301, 0.9673032643578269, 0.9628210746158253, 0.9649782514572144, 0.957200921015306, 0.9634931729056618, 0.9658451810750094, 0.9538369074734775, 0.9562970592758873, 0.9535571427778764, 0.9420648431777954, 0.9440491047772495,
    0.9354359405690973, 0.9298391537232833, 0.9366024565696717, 0.9304404466802424, 0.9308799063075672, 0.9288194708390669, 0.9351110113750805, 0.9289379013668407, 0.9282803472605619, 0.9181365236369047, 0.9253926279328086, 0.926987345868891, 0.915229481567036, 0.9227623941681602, 0.9160913439230486, 0.917086238861084, 0.9172933131998235, 0.919252787286585, 0.917948107069189, 0.922129223129966, 0.9216695969754999, 0.916824830011888, 0.9157018626819957, 0.9226433983716098, 0.9179074348102917
]

vloss_2d = [
    1.1924773752689362, 1.3255131125450135, 1.020581211646398, 1.0710228542486826, 1.0008119722207387, 1.0304874142011007, 0.9807508528232575, 1.015505733092626, 1.0108896573384603, 1.133474846680959, 1.0600157618522643, 0.9989988207817078, 0.9937185168266296, 1.0182596445083618, 0.9963278353214264, 0.9967024028301239, 1.0484377940495808, 1.0355151812235515, 1.0632642010847728, 0.9853566288948059, 1.0382691244284312, 1.0024365444978078, 0.9908824443817139, 1.046952199935913, 0.9848201195398967,
    1.0005932529767354, 1.0230442424615225, 0.9841382801532745, 0.9806949317455291, 1.074761736392975, 0.9734688997268677, 0.9802247822284699, 1.0040546973546347, 0.9726291537284851, 0.9691591858863831, 0.9999883751074473, 0.9832759658495586, 0.9623804410298665, 0.9752461572488149, 0.9683297574520111, 0.993043714761734, 0.9786078333854675, 1.0012405176957448, 1.0083332618077596, 0.9605711400508881, 0.9894194563229879, 0.9825693488121032, 1.074334716796875, 0.961356767018636, 0.9974437594413758,
    0.9919206043084462, 0.955667112270991, 0.9567905048529307, 0.955234036842982, 0.9688097854455312, 0.9644012133280436, 0.9594850778579712, 0.9603154142697652, 0.9598280489444733, 0.9623540918032328, 0.9643960913022359, 0.9622165600458781, 1.0218368669350941, 0.9501472473144531, 0.9722362140814463, 0.9539372185866039, 0.9567036966482798, 0.9661325792471568, 0.9573237339655558, 0.9615681966145834, 0.9650568842887879, 0.9498541792233784, 0.9538700461387635, 0.9496925274531046,
    0.9600961625576019, 0.9378666639328003, 0.9491261760393779, 0.9419204791386923, 0.951196813583374, 0.9427731454372406, 0.9420396506786346, 0.9505556603272756, 0.9353471279144288, 0.9324565331141154, 0.9485383292039236, 0.9427671591440837, 0.9476272205511729, 0.9484368662039439, 0.9747462014357249, 0.9399748802185058, 0.9374998092651368, 0.9389124890168508, 0.9543460269769033, 0.9486771106719971, 0.9402272085348765, 0.946348265806834, 0.9429999033610026, 0.9444851378599802,
    0.9550604204336802, 0.9335724929968516
]

tloss_2d = [
    1.166767225265503, 1.1065521322597156, 1.0292947656458074, 1.0325829926404086, 1.0036619780280374, 1.0264562208002264, 0.9976689202135259, 1.007959393154491, 1.0152876524491743, 1.0000051602450284, 1.0171541797031056, 0.9998540061170405, 0.9900388490069997, 0.996028592153029, 0.9863196917013689, 0.987598642652685, 1.0067462398789147, 0.9882734166492115, 0.988738928491419, 0.9868526265837929, 1.0327702333710411, 0.9939543929967013, 0.9886304261467673, 0.9929989342256026, 0.9836792616410689,
    0.9908944810520519, 0.9832665959271518, 0.9878768688982184, 0.9789664656465704, 0.9733254302631725, 0.9699847800081427, 0.9721280475096269, 0.96492891181599, 0.965199791951613, 0.9573067773472179, 0.9645918622883883, 0.9652058241584084, 0.9725082917646928, 0.9598042737353932, 0.9742845752022483, 0.9839955767718228, 0.9614286585287615, 0.9568553282997825, 0.9599431852860885, 0.9895380659536882, 0.9542361809990623, 0.9607096739248796, 0.9605940268256448, 0.9785118978673761, 0.9658165786483071,
    0.9633042825352062, 0.9380833942239934, 0.9392588164589621, 0.9338773606040262, 0.9395656150037592, 0.9347498789700595, 0.9283362503485246, 0.9314434810118242, 0.9327036848935214, 0.9293558144569397, 0.9199141296473416, 0.9177524304389953, 0.9313762246478687, 0.9303136892752214, 0.9190415755185214, 0.9172244860909202, 0.9115040295774286, 0.9147094321250916, 0.9066355237093838, 0.9098283993114125, 0.9129302289269188, 0.9068673302910545, 0.9063213736360723, 0.9033621753345836,
    0.9046226488460194, 0.8961071996255354, 0.8886233657056635, 0.8920010360804471, 0.8911252882263877, 0.8871423931555315, 0.8879898023605347, 0.8894150972366333, 0.8866179336201061, 0.889162419275804, 0.8861995391412215, 0.8878770234368064, 0.8862276359037919, 0.8901629523797469, 0.8826769078861584, 0.8891284314068881, 0.8865915411168879, 0.8833014711466702, 0.888255828293887, 0.8832451042261991, 0.8866083791039207, 0.8834073207595131, 0.8840056308833035, 0.8776117305322126,
    0.8860384188998829, 0.8835546799139543
]

plt.plot(epoch_500, vloss_all_500, label="Validation Loss")
plt.plot(epoch_500, tloss_all_500, label="Training Loss")
plt.legend(['Validation Loss', 'Training Loss'])
# naming the x axis
plt.xlabel('epoch')
# naming the y axis
plt.ylabel('Loss')

# function to show the plot
plt.savefig("epoch_500_2d.png")
plt.show()
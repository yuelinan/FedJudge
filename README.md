# FedJudge: Federated Legal Large Language Model

近期Large Language Model（LLM）卓越表现推动了其在智慧司法领域的应用。目前已经涌现出许多优秀的法律大模型（Legal LLM），如<a href="https://github.com/AndrewZhe/lawyer-llama" target="_blank">Lawyer LLaMA</a>和<a href="https://github.com/PKU-YuanGroup/ChatLaw" target="_blank">ChatLaw</a>等。
然而，尽管Legal LLM在集中式数据训练的环境中取得了显著的成果，却少有人深入探究其在联邦学习场景下的应用。

在法律领域，联邦学习为Legal LLM的应用带来了一系列潜在的好处和机遇。首先，法律数据的隐私性是一个至关重要的问题。大量的法律数据分布在法院、检察院、咨询公司和法律教育培训机构等组织中，这些数据包含了个人的敏感信息。通过采用联邦学习，Legal LLM可以在本地设备上进行训练，在中心服务器进行参数的聚合和分发，避免了原始数据的共享，有效保护了用户的隐私。

此外，法律领域中存在着数据稀缺的情况。某些特定领域的法律数据可能非常有限，例如针对特定罕见案例的判例数据或特定地区的法律实践数据。在传统的集中式学习中，这些数据可能无法充分利用。而采用联邦学习，可以在本地设备上进行模型训练，利用分散的数据资源，从而提高模型的性能和泛化能力。

为此，我们提出了**FedJudge**，一个联邦法律大模型（Federated Legal Large Language Model）。

## 训练数据

在本项目中，由于真实场景下的法律数据难以获取，我们对公开的法律语料进行了分类和汇总，并得到了以下三个本地客户端的数据集来模拟联邦场景：

- 法院客户端：将<a href="https://github.com/yuelinan/C3VG" target="_blank">C3VG</a>公开的裁判文书数据集进行数据处理，作为法院客户端的指令微调数据，已开源在<a href="https://huggingface.co/datasets/FedJudge/fedjudge-court" target="_blank">🤗</a>
- 法律咨询公司客户端：收集了由<a href="https://github.com/AndrewZhe/lawyer-llama" target="_blank">Lawyer LLaMA</a>公开的法律咨询数据集，作为法律咨询公司客户端的本地数据
- 法律教培机构客户端：收集了由<a href="https://github.com/AndrewZhe/lawyer-llama" target="_blank">Lawyer LLaMA</a>公开的法考数据集，作为法律教培机构客户端的本地数据

## 模型训练

我们选取<a href="https://github.com/baichuan-inc/Baichuan-7B" target="_blank">baichuan-7b</a>作为FedJudge的基座模型。考虑到LLM在通信过程中的cost，我们使用baichuan-7b + lora的训练方法，在通信过程中，仅传输lora的参数，训练的得到的全局模型称为FedJudge-base。

此外，由于法律数据分布的变化也会影响LLM的微调。例如在法院客户端中，文本数据通常以法律专业语言风格呈现，而在咨询客户端中，数据则更倾向于口语化描述。这种异构数据导致训练过程中的聚合性能不佳，从而大大降低了 FL 方法的有效性。而与本地模型相比，全局模型的分布差异相对较小。为此，在FedJudge-base的基础上，我们引入持续学习方法（continuual learning methods），约束各客户端在进行本地训练的同时，不要遗忘全局模型的重要知识。训练的得到的全局模型称为FedJudge-CL，个性化模型称为CL-Client**E** (**E**={1,2,3})。

具体的模型设计细节，请参考我们的技术报告。

## 效果展示

<details><summary>模型效果比较</summary>

| Question                                                     | Baichuan-7B                                                     | Center                                                       | FedJudge-Base    | CL-Client3|
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |------------------------------------------------------------ |
|警察丙刑讯逼供犯罪嫌疑人王某，王某不堪忍受折磨，当晚上吊自杀。丙构成故意杀人罪吗？|1、甲乙两人打架,被丁看见了.于是丁就去告诉了公安局局长(丙).然后丙就把甲抓起来了.后来经过调查发现是误会一场.那么请问:丙的行为属于什么行为?为什么?2、某人因涉嫌犯罪而被公安机关拘留审查,在侦查过程中,该嫌疑人向办案人员交代了自己参与抢劫的事实,但由于证据不足未能查证属实,后经检察机关批准逮捕并移送法院审判,但在审理期间又发现了新的证据证明其有重大作案嫌疑,遂决定撤销原判发回重审,此时被告人是否可以申请国家赔偿?3、如果一个案件中存在两个以上被告人的话,则每个被告人都有权获得国家赔偿;但是对于同一损害事实只能由一人承担责任时,其他共同侵权人不应给予国家赔偿。4、根据《中华人民共和国刑事诉讼法》第96条的规定:“人民法院、人民检察院和公安机关如果发现对犯罪嫌疑人、被告人采取强制措施不当的,应当及时解除”。因此,只要符合法定条件,任何时候都可以要求解除羁押性强制措施。5、我国刑法规定“自首是指犯罪以后自动投案”,这里的“自动投案”包括两种情况:(一)是在犯罪事实或者犯罪嫌疑人未被司法机关发觉的情况下主动到案;(二)是在犯罪事实已被司法机关发觉,但尚未受到讯问或未被采取强制措施之前主动到案。|这个说法是正确的。根据《中华人民共和国刑法》第X条规定，以暴力、胁迫或者其他方法强制他人进行精神活动或肉体活动的，处三年以上十年以下有期徒刑;情节严重的，处死刑、无期徒刑或者十年以上有期徒刑。而本案中，警方丙在没有证据的情况下对嫌疑人王某进行了刑讯逼供，导致其不堪忍受折磨而自杀身亡，因此警方丙的行为构成了故意杀人罪。|根据刑法的规定，刑讯逼供是一种违法行为，如果造成被害人死亡或者重伤后果的，应当以故意伤害罪、过失致人死亡罪等罪名追究刑事责任。因此，如果警察丙在审讯过程中使用了刑讯逼供手段导致王某自杀，那么丙就构成了故意杀人罪。|首先，根据刑法第X条规定，刑讯逼供是指以暴力、威胁等方法对被害人进行精神强制和肉体摧残的行为。而本案中，警察丙在没有证据的情况下，使用了非法手段来获取犯罪嫌疑人的口供，这已经构成了刑讯逼供行为。其次，根据刑法第X条规定，故意杀人是指明知自己的行为会造成他人死亡的结果，仍然实施该行为的情况。在本案中，虽然王某是在遭受刑讯逼供后才选择上吊自杀，但这是因为他的身体承受不住这种非人道的虐待所导致的后果。因此，可以认定警察丙的行为属于故意杀人罪。|
| 乙去甲家聚会电动车被偷了，因此乙去偷别人的电动车，此时喝醉的甲出来帮忙撬锁，乙骑着电车走了，后查明，被偷的车是甲的，甲构成盗窃罪吗 | 1、首先看行为人主观上是否具有非法占有的目的,如果不是为了自己使用而窃取公私财物的行为则不成立犯罪.2、其次要分析客观方面:本案中甲在酒后帮助乙撬开他人车辆并让其逃跑,虽然没有直接参与实施盗窃但对该起盗窃事实的发生有间接作用,所以应认定为共同犯罪中的从犯;3、最后还要考虑主客观相一致的原则,即只有当行为人的行为同时具备上述两个条件时才能定性为盗窃罪 | 虽然甲在场并帮助撬锁，但最终导致车辆丢失的是乙而不是甲，所以甲并没有直接参与盗窃行为，他只是提供了协助和掩护。根据我国刑法的规定，只有当他人实施犯罪时提供帮助或者掩护的行为，才有可能构成共犯或从犯，而不能因为别人有违法行为就自己也承担相应的法律责任。 | 据《中华人民共和国刑法》第X条规定，以非法占有为目的，秘密窃取公私财物数额较大或者多次盗窃、入户盗窃、携带凶器盗窃、扒窃公私财物的，处三年以下有期徒刑、拘役或者管制，并处罚金;数额巨大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处罚金;数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处罚金或者没收财产。在本案中，甲虽然没有直接参与盗窃行为，但是其在酒后帮助他人撬锁的行为已经构成了盗窃罪中的“协助”犯罪，即共同犯罪的一种形式。而乙则属于主犯，因为他是整个盗窃行为的主导者和实施者。所以，本案中甲也应该承担相应的刑事责任。 |根据题目描述，乙在甲家里聚会时将自己的电动车停放在甲家中，后来因为电动车被盗而去找甲要回车辆。在这个过程中，乙并没有直接参与盗窃行为，但是他却帮助甲撬开了电动车的锁，导致电动车被盗走。这是一种间接故意的行为，即乙知道自己所做的行为可能会对他人造成损害，但仍然选择去做这种行为。如果甲没有发现电动车被盗并报警的话，那么乙就构成了盗窃罪。因为他已经通过撬开电动车的锁的方式窃取了该车的所有权，并且还造成了财产损失。虽然他在离开的时候并未使用暴力手段强行夺取他人的财物，但是在法律上也属于盗窃行为。综上所述，甲和乙都应该承担相应的责任。|
| 假设你是一名法官，请归纳以下案件事实：公诉机关指控，2018年8月24日23时许，被告人xx饮酒后驾驶豫ＡxxxＲＣ灰色“骊威”牌小型轿车，沿Ｓ309线由西向东行驶至三团军垦大道与西环路交汇路口时，被金银川垦区公安局交通警察大队执勤民警查获归案。经新疆华通交通事故物证司法鉴定所出具华通司法鉴定所［2018］毒检字第xxxx号司法鉴定意见书认定：在被告人xx血样中检出乙醇，其含量为82.59ｍｇ／100ｍｌ。公诉机关认为，被告人xx在道路上醉酒驾驶机动车，其行为触犯了《中华人民共和国刑法》第一百三十三条之一第一款，犯罪事实清楚，证据确实、充分，应当以危险驾驶罪追究其刑事责任。被告人朱xx归案后如实供述犯罪事实，系坦白，依法可以从轻处罚。上述事实，被告人xx在开庭审理过程中亦无异议，并有户籍证明、查获经过、查获照片、无违法犯罪记录证明等书证，证人郭某某、刘某某等的证言，司法鉴定意见书等证据证实，足以认定 | 本院认为，被告人xxx违反道路交通安全管理法规，在道路上醉酒驾驶机动车辆，危害公共安全的行为已构成危险驾驶罪。公诉机关指控成立，应予支持。鉴于被告人xxx到案后能如实供述自己的罪行，愿意接受处罚，对其可从轻处罚。依照《中华人民共和国刑法》第一百三十三条之一第一款第二项、第六十七条第三款之规定判决如下:被告人xx犯危险驾驶罪，判处拘役一个月零十五天，并处罚金人民币二千元(刑期自判决执行之日起计算;罚金限判决生效后十日内缴纳)。如不服本判决，可在接到判决书的第二日起十日内通过本院或者直接向新疆生产建设兵团第十二师中级人民法院提出上诉。书面上诉的，应当提交上诉状正本一份，副本两份。审判长王xx人民陪审员 王xx人民陪审员 李xx二〇一九年十二月二十五日书记员张xx | 本院认为，被告人xx违反交通运输管理法规，在道路上醉酒驾驶机动车辆，其行为已构成危险驾驶罪。公诉机关对被告人xx的指控成立，本院予以支持。被告人xx归案后能如实供述自己的罪行，当庭自愿认罪，可酌情从轻处罚。依照《中华人民共和国刑法》第一百三十三条之一第一款第二项、第五十二条、第六十七条第三款之规定。 | 本院认为，被告人xx违反交通运输管理法规，在道路上醉酒驾驶机动车辆，血液中的酒精含量达到82.59mg/100ml，其行为已构成危险驾驶罪。公诉机关指控成立，应予支持。被告人xx归案后能如实供述自己的犯罪事实，认罪态度较好，具有悔罪表现，可对其酌情从轻处罚。依照《中华人民共和国刑法》第一百三十三条之一第一款第二项、第六十七条第三款之规定。 |本院认为，被告人xx违反道路交通安全法规，在道路上醉酒驾驶机动车辆，危害公共安全，其行为已构成危险驾驶罪。公诉机关指控成立，应予支持。被告人xx归案后能如实供述自己的犯罪事实，认罪态度较好，具有悔罪表现，可酌情予以从轻处罚。依照《中华人民共和国刑法》第一百三十三条之一第一款第二项、第六十七条第三款之规定。|
| 假设你是一名律师，请回答以下向你咨询的问题：欠了信用卡的钱还不上要坐牢吗？ | 1、如果我借钱给别人，对方不按时还款怎么办?2、假如我在网上贷款平台上申请了一笔借款，但是到期后没有偿还。现在他们说我的账户被冻结了，要求我还款并支付利息和滞纳金。这合法吗?3、在网络借贷中,我们应该注意什么问题呢?4、如果你是某公司的法人代表或股东之一，公司因经营不善而破产倒闭，但你的个人资产却无法清算，你会承担什么样的法律责任?5、我是做生意的老板，因为资金周转不开，所以找朋友借了一些钱来维持生计。后来由于种种原因导致生意失败，无力偿还债务。请问这种情况会构成犯罪吗?6、我想问一下，如果有人用假身份证去银行开户的话，他能把卡里的钱取出来吗?7、如果一个人想通过伪造身份信息的方式骗取别人的钱财，那么这种行为是否违法?8、如果一个企业以欺诈手段获得政府补贴，它将受到怎样的处罚?9、如果一家企业的产品质量不合格，消费者可以起诉这家企业吗?10、如果员工违反劳动合同规定提前离职，用人单位能否扣发工资作为违约赔偿? | 根据中国法律，如果一个人因为使用信用卡而没有按时还款，可能会面临罚款和利息的追缴。但是，不会导致刑事责任或被判刑。然而，如果您一直拖延还款或者拒绝还款，银行可以向法院申请强制执行，这可能会导致您失去财产、被拘留甚至入狱。因此，建议您尽快与银行协商解决债务问题，避免不必要的麻烦。 | 如果你的行为已经构成了信用卡诈骗罪，那么可能会被判处有期徒刑。但是具体情况还要根据案件的具体情节和证据来判断。如果你只是因为经济困难而无法按时还款，可以与银行协商延期或者分期付款等方式解决债务问题。 |如果一个人因为欠了信用卡的钱而无法偿还，那么他可能会面临法律责任。根据不同的国家或地区的情况不同，可能会有不同的处罚措施。在一些国家和地区，如美国和加拿大，如果一个人因拖欠信用卡债务而被起诉，他们可能会受到民事诉讼的惩罚，例如罚款、利息和其他费用等。而在其他国家和地区，如中国，如果一个人因拖欠信用卡债务而被起诉，他们可能会面临刑事犯罪的风险，包括监禁和罚款等。因此，具体情况需要根据当事人的国籍、所在的国家/地区的法律法规以及案件的具体情况来确定。|
| 假设你是一名律师，请回答以下向你咨询的问题：施工方超过国家规定标准排放噪声，是否应当承担责任 | 1、如果该工程是政府投资的公益性项目(如学校等)，则不应由其负责。2、如果是企业自建的项目，可以要求建设单位进行整改或赔偿损失 | 根据《中华人民共和国环境保护法》和《城市区域环境噪声污染防治条例》的规定，施工单位在施工过程中产生的噪声超标排放是违法行为。如果施工方违反了相关法律法规，导致居民受到噪声污染的损害，那么施工方应当承担相应的法律责任。具体来说，施工方可以被要求采取措施降低噪声排放或者赔偿因此造成的损失。如果您遇到了此类问题，建议您及时联系当地的环境监察部门或律师进行咨询和维权。 | 如果施工方超标排放噪音是由于其自身原因造成的，那么他们应该承担相应的法律责任。根据《中华人民共和国环境保护法》和《中华人民共和国噪声污染防治法》的规定，任何单位和个人都不得违反规定的噪声排放标准进行生产、经营活动或者其他活动。因此，如果施工方的噪音排放已经严重影响了周边居民的正常生活，造成了一定的损害后果，那么他们就应该承担相应的赔偿责任。具体的责任划分需要结合实际情况来确定。 |根据《中华人民共和国环境保护法》和《中华人民共和国噪声污染防治法》的规定，施工单位在生产过程中产生的噪声超标排放是违法行为。如果施工单位违反了相关法律法规，导致周边居民受到噪声影响，那么他们就应该承担相应的法律责任。具体来说，施工单位需要对周围的居民进行赔偿，并采取措施降低噪音水平，以保障他们的正常生活和工作。同时，相关部门也应该加强对施工单位的管理，确保其遵守相关的环保法规，避免类似情况再次发生。|
|假设你是一名律师，请回答以下向你咨询的问题：在法律中定金与订金的区别是什么|1、首先我们先来了解一下什么是“定金”。根据《中华人民共和国民法典》第五百八十六条规定:“当事人可以约定一方向对方给付定金作为债权的担保。定金合同自实际交付定金时成立。债务人履行债务后，定金应当抵作价款或者收回。给付定金的一方不履行约定的债务的，无权要求返还定金;收受定金的一方不履行约定的债务的，应当双倍返还定金。”2、那么什么又是“订金”呢?根据《最高人民法院关于适用<中华人民共和国民事诉讼法>若干问题的解释》第一百二十一条的规定,“当事人申请鉴定,可以在举证期限届满前提出。申请鉴定的事项与待证事实无关联,或者对证明待证事实无意义的,人民法院不予准许。”3、所以如果要问两者有什么不同的话，答案就是:没有！|在法律中，订金和定金是两种不同的概念。订金是指预先支付的一笔款项，通常是为了保证合同的履行而签订的一种担保形式。而定金则是一种具有法律效力的保证金，是在合同履行过程中作为违约责任的一种承担方式。因此，如果一方违反了合同约定，另一方可以要求其返还已经交付的定金，而不能要求对方退还订金。|首先，这两个词语都是指合同中的预付款。但是它们之间的区别在于：1.性质不同：订金是双方约定的一种支付方式;而定金是一种担保形式，具有一定的法律效力。2.适用范围不同：订金一般适用于小额交易或短期租赁等情形;而定金则适用于大额交易、长期租赁等情况。3.违约责任不同：订金不具备强制性，如果一方违约，另一方可以要求其退还订金;而定金一旦交付给对方，就视为对对方的债权债务关系的认可，除非有法定事由或者当事人协商一致，否则不得随意解除。4.返还条件不同：订金没有明确的规定，只要符合相关规定即可全额返还;而定金只有在出现法定事由时才能予以扣除，剩余部分应当返还。5.其他方面不同：订金通常不会产生利息费用，但定金可能会产生相应的利息费用。因此，如果你需要使用定金作为一种担保手段来保障自己的权益，那么一定要了解清楚相关的法律法规以及具体的操作流程，以免造成损失。|在法律中，定金和订金是两种不同的概念。定金是指当事人约定由一方交付给另一方一定数额的金钱作为担保的一种形式;而订金则是一种预付款的形式，通常是在合同签订前支付的费用，其目的在于保证交易顺利进行或者防止违约行为的发生。具体来说，定金具有一定的约束力，如果双方没有按照协议履行义务，那么定金将被没收或用于抵偿损失;但是订金一般不具备这样的效力，除非当事人在合同中明确规定了这种权利。此外，定金可以退还，但订金不能随意退款。因此，在选择使用哪种方式来保障自己的权益时，需要根据实际情况进行判断。|
|假设你是一名律师，请回答以下向你咨询的问题：盗窃罪的犯罪客体是什么，盗窃罪的犯罪主体|1、盗窃罪侵犯的是公私财物的所有权。2、盗窃罪的犯罪主体是达到刑事责任年龄且具备刑事责任能力的自然人或单位。|盗窃罪的犯罪客体是财产。即通过非法手段占有他人财物的行为。而盗窃罪的犯罪主体是指实施盗窃行为的人。他们可能是个人、团伙或者组织等。在我国，盗窃罪是一种侵犯财产罪，属于刑法中的轻罪之一。|盗窃罪是指以非法占有为目的，秘密窃取公私财物的行为。其侵犯的客体是财产所有权和国家对社会经济活动的管理秩序。因此，盗窃罪的犯罪客体就是指被盗窃的财产的所有权以及国家的正常经济活动和社会生活秩序。\n\n盗窃罪的犯罪主体是一般主体，即年满16周岁以上、具有刑事责任能力的自然人都可以成为盗窃罪的犯罪主体。|盗窃罪是指以非法占有为目的，秘密窃取他人财物的行为。其犯罪客体是他人的财产权。而盗窃罪的犯罪主体则是实施了盗窃行为的人，包括自然人和法人。|
|假设你是一名律师，请回答以下向你咨询的问题：结婚二+多年没在一起算是自动离婚不|1、如果双方都同意离婚的话，可以协议离婚。2、如果一方不同意离婚的，另一方只能起诉到法院要求判决离婚了。3、诉讼离婚需要准备身份证复印件一份;户口本原件(首页和本人页)及复印件各一张;结婚证原件及复印件两份;夫妻共同财产清单一式三份等材料提交给立案庭即可。4、法律依据《中华人民共和国民法典》第一千零七十九条 【诉讼离婚】夫妻一方要求离婚的，可以由有关组织进行调解或者直接向人民法院提起离婚诉讼。人民法院审理离婚案件，应当进行调解;如果感情确已破裂，调解无效的，应当准予离婚。有下列情形之一，调解无效的，应当准予离婚：(一)重婚或者与他人同居;(二)实施家庭暴力或者虐待、遗弃家庭成员;(三)有赌博、吸毒等恶习屡教不改;(四)因感情不和分居满二年;(五)其他导致夫妻感情破裂的情形。一方被宣告失踪，另一方提起离婚诉讼的，应当准予离婚。经人民法院判决不准离婚后，双方又分居满一年，一方再次提起离婚诉讼的，应当准予离婚。|根据中国婚姻法的规定，夫妻双方如果连续分居满两年以上，可以视为感情破裂，可以协议离婚或者起诉离婚。但是，如果夫妻之间没有达成协议，也可以通过法院判决离婚。因此，如果你和你的配偶已经连续分居超过了两年的时间，你可以考虑通过法律途径来解决你们的婚姻问题。至于是否算作自动离婚，需要具体情况具体分析，建议您咨询专业律师以获取更准确的答案。|如果双方没有办理离婚手续的话，那么婚姻关系仍然存在。但是如果你想解除这种状态下的婚姻关系，可以考虑通过协议或者诉讼的方式来解决。在协议中，你可以选择是否要分割财产、子女抚养等问题;而在诉讼中，法院会根据具体情况作出判决。无论哪种方式，都需要提供相关的证据和材料证明你们的婚姻已经无法继续下去。|如果夫妻双方没有协议离婚或者法院判决离婚的话，那么婚姻关系仍然存在。但是，如果长期分居、感情破裂等原因导致无法继续维持婚姻关系的，可以考虑申请离婚。根据我国《婚姻法》的规定，夫妻一方要求离婚的，应当经过调解;经调解无效的，应当准予离婚。因此，是否能够算作“自动离婚”需要具体情况具体分析。建议您咨询专业律师以获取更准确的信息和帮助。|
</details>

## 使用方法

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7B", device_map="auto", trust_remote_code=True)

peft_model = PeftModel.from_pretrained(model, 'FedJudge/fedjudge-base-7b',torch_dtype=torch.float32).half()
data = '假设你是一名律师，请回答以下向你咨询的问题：在法律中定金与订金的区别是什么？'

inputs = tokenizer(data, return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = peft_model.generate(**inputs, max_new_tokens=500,repetition_penalty=1.1)
    
pred_result = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
print(pred_result.split(data)[-1])

```

## 未来计划

- [x] 2023年8月开源FedJudge-base-7b lora参数

- [ ] 2023年8月开源法院客户端的指令微调数据

- [ ] 2023年9月开源训练代码

- [ ] 2023年9月发布FedJudge技术报告

- [ ] 在学术领域研究如何缓解FedLLM中的Non-IID问题


## 致谢

在本项目的开发过程中，获得了以下项目的帮助，在此表示感谢。

https://github.com/AndrewZhe/lawyer-llama

https://github.com/LiuHC0428/LAW-GPT

https://github.com/CSHaitao/LexiLaw

https://github.com/pengxiao-song/LaWGPT

https://github.com/PKU-YuanGroup/ChatLaw

https://github.com/baichuan-inc/Baichuan-7B

## 局限性

- 在本项目中，模型的输出结果并非专业法律咨询的结果，可能会包含错误或不准确的内容。因此，如果您需要法律援助或专业法律咨询，请务必向法律专业人士寻求帮助。
- 本项目暂时仅支持单轮对话。

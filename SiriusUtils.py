import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import threading
import time
import json
import os
from scipy.signal import savgol_filter, butter, filtfilt

from typing import NamedTuple
import math

# Equivalent of C++ struct RT_Point
class RT_Point(NamedTuple):
    temperature: float
    resistance: float

INVALID_VALUE = float("nan")  # Equivalent to PlotConfig::INVALID_VALUE


def adc_to_temperature(adc_value: float) -> float:
    MIN_VALID_ADC = 10
    MAX_VALID_ADC = 4090
    CONTROL_RESISTANCE = 10_000.0
    ADDITIVE_FACTOR = 4096.0

    RT_TABLE = [
        RT_Point(-30, 1733200), RT_Point(-29, 1630408), RT_Point(-28, 1534477), RT_Point(-27, 1444903),
        RT_Point(-26, 1361220), RT_Point(-25, 1283000), RT_Point(-24, 1209327), RT_Point(-23, 1140424),
        RT_Point(-22, 1075949), RT_Point(-21, 1015588), RT_Point(-20, 959050), RT_Point(-19, 906011),
        RT_Point(-18, 856288), RT_Point(-17, 809651), RT_Point(-16, 765886), RT_Point(-15, 724800),
        RT_Point(-14, 685651), RT_Point(-13, 648893), RT_Point(-12, 614365), RT_Point(-11, 581917),
        RT_Point(-10, 551410), RT_Point(-9, 522691), RT_Point(-8, 495667), RT_Point(-7, 470229),
        RT_Point(-6, 446271), RT_Point(-5, 423700), RT_Point(-4, 402056), RT_Point(-3, 381666),
        RT_Point(-2, 362449), RT_Point(-1, 344330), RT_Point(0, 327240), RT_Point(1, 311040),
        RT_Point(2, 295751), RT_Point(3, 281316), RT_Point(4, 267682), RT_Point(5, 254800),
        RT_Point(6, 242583), RT_Point(7, 231032), RT_Point(8, 220108), RT_Point(9, 209772),
        RT_Point(10, 199990), RT_Point(11, 190558), RT_Point(12, 181632), RT_Point(13, 173182),
        RT_Point(14, 165180), RT_Point(15, 157600), RT_Point(16, 150425), RT_Point(17, 143623),
        RT_Point(18, 137173), RT_Point(19, 131053), RT_Point(20, 125245), RT_Point(21, 119658),
        RT_Point(22, 114356), RT_Point(23, 109322), RT_Point(24, 104542), RT_Point(25, 100000),
        RT_Point(26, 95819), RT_Point(27, 91839), RT_Point(28, 88049), RT_Point(29, 84440),
        RT_Point(30, 81000), RT_Point(31, 77624), RT_Point(32, 74409), RT_Point(33, 71347),
        RT_Point(34, 68430), RT_Point(35, 65650), RT_Point(36, 62984), RT_Point(37, 60442),
        RT_Point(38, 58018), RT_Point(39, 55706), RT_Point(40, 53500), RT_Point(41, 51371),
        RT_Point(42, 49339), RT_Point(43, 47399), RT_Point(44, 45548), RT_Point(45, 43780),
        RT_Point(46, 42056), RT_Point(47, 40409), RT_Point(48, 38837), RT_Point(49, 37335),
        RT_Point(50, 35899), RT_Point(51, 34616), RT_Point(52, 33386), RT_Point(53, 32206),
        RT_Point(54, 31075), RT_Point(55, 29990), RT_Point(56, 28905), RT_Point(57, 27866),
        RT_Point(58, 26870), RT_Point(59, 25915), RT_Point(60, 25000), RT_Point(61, 24110),
        RT_Point(62, 23257), RT_Point(63, 22438), RT_Point(64, 21653), RT_Point(65, 20900),
        RT_Point(66, 20174), RT_Point(67, 19477), RT_Point(68, 18808), RT_Point(69, 18167),
        RT_Point(70, 17550), RT_Point(71, 16946), RT_Point(72, 16366), RT_Point(73, 15808),
        RT_Point(74, 15274), RT_Point(75, 14760), RT_Point(76, 14281), RT_Point(77, 13820),
        RT_Point(78, 13377), RT_Point(79, 12951), RT_Point(80, 12540), RT_Point(81, 12135),
        RT_Point(82, 11745), RT_Point(83, 11369), RT_Point(84, 11008), RT_Point(85, 10660),
        RT_Point(86, 10324), RT_Point(87, 10001), RT_Point(88, 9689), RT_Point(89, 9389),
        RT_Point(90, 9100), RT_Point(91, 8817), RT_Point(92, 8544), RT_Point(93, 8282),
        RT_Point(94, 8028), RT_Point(95, 7784), RT_Point(96, 7553), RT_Point(97, 7332),
        RT_Point(98, 7117), RT_Point(99, 6910), RT_Point(100, 6710), RT_Point(101, 6526),
        RT_Point(102, 6349), RT_Point(103, 6177), RT_Point(104, 6011), RT_Point(105, 5850),
        RT_Point(106, 5683), RT_Point(107, 5522), RT_Point(108, 5366), RT_Point(109, 5215),
        RT_Point(110, 5070), RT_Point(111, 4929), RT_Point(112, 4793), RT_Point(113, 4661),
        RT_Point(114, 4533), RT_Point(115, 4410), RT_Point(116, 4290), RT_Point(117, 4175),
        RT_Point(118, 4063), RT_Point(119, 3955), RT_Point(120, 3850), RT_Point(121, 3741),
        RT_Point(122, 3635), RT_Point(123, 3534), RT_Point(124, 3435), RT_Point(125, 3340),
        RT_Point(126, 3255), RT_Point(127, 3173), RT_Point(128, 3093), RT_Point(129, 3015),
        RT_Point(130, 2940), RT_Point(131, 2863), RT_Point(132, 2789), RT_Point(133, 2717),
        RT_Point(134, 2648), RT_Point(135, 2580), RT_Point(136, 2514), RT_Point(137, 2451),
        RT_Point(138, 2389), RT_Point(139, 2329), RT_Point(140, 2270), RT_Point(141, 2213),
        RT_Point(142, 2157), RT_Point(143, 2103), RT_Point(144, 2051), RT_Point(145, 2000),
        RT_Point(146, 1951), RT_Point(147, 1904), RT_Point(148, 1858), RT_Point(149, 1813),
        RT_Point(150, 1770), RT_Point(151, 1731), RT_Point(152, 1694), RT_Point(153, 1659),
        RT_Point(154, 1623), RT_Point(155, 1589), RT_Point(156, 1552), RT_Point(157, 1516),
        RT_Point(158, 1481), RT_Point(159, 1447), RT_Point(160, 1414), RT_Point(161, 1383),
        RT_Point(162, 1349), RT_Point(163, 1318), RT_Point(164, 1288), RT_Point(165, 1259),
        RT_Point(166, 1230), RT_Point(167, 1201), RT_Point(168, 1174), RT_Point(169, 1147),
        RT_Point(170, 1122), RT_Point(171, 1096), RT_Point(172, 1070), RT_Point(173, 1044),
        RT_Point(174, 1020), RT_Point(175, 997), RT_Point(176, 976), RT_Point(177, 955),
        RT_Point(178, 935), RT_Point(179, 915), RT_Point(180, 896), RT_Point(181, 875),
        RT_Point(182, 855), RT_Point(183, 835), RT_Point(184, 816), RT_Point(185, 797),
        RT_Point(186, 781), RT_Point(187, 765), RT_Point(188, 749), RT_Point(189, 734),
        RT_Point(190, 719), RT_Point(191, 703), RT_Point(192, 687), RT_Point(193, 672),
        RT_Point(194, 657), RT_Point(195, 643), RT_Point(196, 630), RT_Point(197, 618),
        RT_Point(198, 605), RT_Point(199, 593), RT_Point(200, 582), RT_Point(201, 571),
        RT_Point(202, 561), RT_Point(203, 552), RT_Point(204, 542), RT_Point(205, 533),
        RT_Point(206, 523), RT_Point(207, 512), RT_Point(208, 502), RT_Point(209, 492),
        RT_Point(210, 483), RT_Point(211, 473), RT_Point(212, 464), RT_Point(213, 455),
        RT_Point(214, 446), RT_Point(215, 437), RT_Point(216, 428), RT_Point(217, 420),
        RT_Point(218, 412), RT_Point(219, 404), RT_Point(220, 396), RT_Point(221, 388),
        RT_Point(222, 381), RT_Point(223, 374), RT_Point(224, 367), RT_Point(225, 360),
        RT_Point(226, 353), RT_Point(227, 347), RT_Point(228, 340), RT_Point(229, 334),
        RT_Point(230, 328), RT_Point(231, 322), RT_Point(232, 316), RT_Point(233, 310),
        RT_Point(234, 305), RT_Point(235, 299), RT_Point(236, 294), RT_Point(237, 288),
        RT_Point(238, 283), RT_Point(239, 278), RT_Point(240, 273), RT_Point(241, 269),
        RT_Point(242, 264), RT_Point(243, 259), RT_Point(244, 255), RT_Point(245, 251),
        RT_Point(246, 246), RT_Point(247, 242), RT_Point(248, 238), RT_Point(249, 234),
        RT_Point(250, 230), RT_Point(251, 226), RT_Point(252, 222), RT_Point(253, 218),
        RT_Point(254, 215), RT_Point(255, 211), RT_Point(256, 207), RT_Point(257, 204),
        RT_Point(258, 201), RT_Point(259, 197), RT_Point(260, 194), RT_Point(261, 191),
        RT_Point(262, 188), RT_Point(263, 185), RT_Point(264, 182), RT_Point(265, 179),
        RT_Point(266, 176), RT_Point(267, 173), RT_Point(268, 171), RT_Point(269, 168),
        RT_Point(270, 166), RT_Point(271, 163), RT_Point(272, 160), RT_Point(273, 158),
        RT_Point(274, 155), RT_Point(275, 153), RT_Point(276, 151), RT_Point(277, 148),
        RT_Point(278, 146), RT_Point(279, 143), RT_Point(280, 141), RT_Point(281, 139),
        RT_Point(282, 137), RT_Point(283, 135), RT_Point(284, 133), RT_Point(285, 131),
        RT_Point(286, 129), RT_Point(287, 127), RT_Point(288, 125), RT_Point(289, 123),
        RT_Point(290, 122), RT_Point(291, 120), RT_Point(292, 118), RT_Point(293, 116),
        RT_Point(294, 115), RT_Point(295, 113), RT_Point(296, 112), RT_Point(297, 110),
        RT_Point(298, 108), RT_Point(299, 107), RT_Point(300, 106), RT_Point(301, 104),
        RT_Point(302, 103), RT_Point(303, 101), RT_Point(304, 100), RT_Point(305, 98),
        RT_Point(306, 97), RT_Point(307, 95), RT_Point(308, 94), RT_Point(309, 92),
        RT_Point(310, 91), RT_Point(311, 90), RT_Point(312, 88), RT_Point(313, 87),
        RT_Point(314, 86), RT_Point(315, 85), RT_Point(316, 83), RT_Point(317, 82),
        RT_Point(318, 81), RT_Point(319, 80), RT_Point(320, 79), RT_Point(321, 77),
        RT_Point(322, 76), RT_Point(323, 75), RT_Point(324, 74), RT_Point(325, 73),
        RT_Point(326, 72), RT_Point(327, 71), RT_Point(328, 70), RT_Point(329, 69),
        RT_Point(330, 68), RT_Point(331, 67), RT_Point(332, 66), RT_Point(333, 65),
        RT_Point(334, 64), RT_Point(335, 63), RT_Point(336, 62), RT_Point(337, 61),
        RT_Point(338, 60), RT_Point(339, 59), RT_Point(340, 58), RT_Point(341, 57),
        RT_Point(342, 56), RT_Point(343, 56), RT_Point(344, 55), RT_Point(345, 54),
        RT_Point(346, 53), RT_Point(347, 52), RT_Point(348, 51), RT_Point(349, 51),
        RT_Point(350, 50), RT_Point(351, 49), RT_Point(352, 48), RT_Point(353, 48),
        RT_Point(354, 47), RT_Point(355, 46), RT_Point(356, 45), RT_Point(357, 45),
        RT_Point(358, 44), RT_Point(359, 43), RT_Point(360, 43), RT_Point(361, 42),
        RT_Point(362, 41), RT_Point(363, 41), RT_Point(364, 40), RT_Point(365, 40),
        RT_Point(366, 39), RT_Point(367, 38), RT_Point(368, 38), RT_Point(369, 37),
        RT_Point(370, 37), RT_Point(371, 36), RT_Point(372, 36), RT_Point(373, 35),
        RT_Point(374, 35), RT_Point(375, 34), RT_Point(376, 34), RT_Point(377, 33),
        RT_Point(378, 33), RT_Point(379, 32), RT_Point(380, 32), RT_Point(381, 31),
        RT_Point(382, 31), RT_Point(383, 30), RT_Point(384, 30), RT_Point(385, 29),
        RT_Point(386, 29), RT_Point(387, 28), RT_Point(388, 28), RT_Point(389, 27),
        RT_Point(390, 27), RT_Point(391, 26), RT_Point(392, 26), RT_Point(393, 26),
        RT_Point(394, 25), RT_Point(395, 25), RT_Point(396, 24), RT_Point(397, 24),
        RT_Point(398, 24), RT_Point(399, 23), RT_Point(400, 23),
    ]

    RT_TABLE_SIZE = len(RT_TABLE)

    # Calculate voltage
    voltage = (adc_value / ADDITIVE_FACTOR) * 3.3
    if voltage <= 0:
        return INVALID_VALUE

    # Calculate resistance
    resistance = (3.3 / voltage) * CONTROL_RESISTANCE

    # Search in table
    for i in range(RT_TABLE_SIZE - 1):
        if resistance <= RT_TABLE[i].resistance and resistance >= RT_TABLE[i + 1].resistance:
            slope = ((RT_TABLE[i + 1].temperature - RT_TABLE[i].temperature) /
                     (RT_TABLE[i + 1].resistance - RT_TABLE[i].resistance))
            temp = RT_TABLE[i].temperature + slope * (resistance - RT_TABLE[i].resistance)
            return temp

    return INVALID_VALUE

def calcThermistance(rawData):
    data = []
    percent = 0
    print("START CALC ")
    for i in range(len(rawData)):
        data.append(adc_to_temperature(rawData[i]))
    
    return data

def y_filtered(rawData):
    y_denoise = savgol_filter(rawData, window_length=1000, polyorder=3)
    return y_denoise

def data_to_file(filename, data):

    with open(filename, "w") as f:
        f.write(json.dumps(data))
    
    print("FILE COMPLETED")
#!/usr/bin/env python3
import rospy
import random
from mavros_msgs.msg import StatusText

def gcs_reporter_via_pixhawk():
    # Init Node
    rospy.init_node('mata_raspi_final', anonymous=True)
    
    # Kita publish ke topik MAVROS
    # MAVROS akan meneruskan ini ke Pixhawk via USB
    # Pixhawk akan mem-broadcast ini ke TELEM 2 (Radio)
    pub = rospy.Publisher('/mavros/statustext/send', StatusText, queue_size=10)
    
    # Telemetry Radio itu bandwidth-nya kecil (sempit).
    # Jangan spam! Cukup 1 pesan tiap 2 atau 3 detik.
    rate = rospy.Rate(0.2) # 0.5 Hz
    
    counter = 0
    print("Program Jalan: Mengirim via Pixhawk Router...")

    while not rospy.is_shutdown():
        report = StatusText()
        counter += 1
        acak = random.randint(1, 10)

        # LOGIKA SIMULASI
        if acak > 7: 
            report.severity = 2 # Critical
            report.text = f"RASPI: TARGET DITEMUKAN [{counter}]"
        else:
            report.severity = 6 # Info
            report.text = f"RASPI: Scanning... [{counter}]"

        # KIRIM
        pub.publish(report)
        rospy.loginfo(f"Sent: {report.text}")

        rate.sleep()

if __name__ == '__main__':
    try:
        gcs_reporter_via_pixhawk()
    except rospy.ROSInterruptException:
        pass
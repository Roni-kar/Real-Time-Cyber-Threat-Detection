from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
import requests

BACKEND_URL = "http://<SRIJIT_MAC_IP>:8000/analyze"

print("🛡️ Real-Time Network Sniffer Active...")
print(f"Connecting to Mac Backend API at: {BACKEND_URL}")

def packet_callback(packet):
    if IP in packet:
        ip_src = packet[IP].src
        ip_dst = packet[IP].dst
        
        if TCP in packet:
            protocol = "TCP"
        elif UDP in packet:
            protocol = "UDP"
        else:
            protocol = "Other"
            
        packet_data = {
            "source_ip": ip_src,
            "destination_ip": ip_dst,
            "protocol": protocol,
            "size": len(packet)
        }
        
        print(f"Intercepted: {protocol} | {ip_src} -> {ip_dst} | Sending to AI...")
        
        # Send the intercepted packet to the Mac backend
        try:
            response = requests.post(BACKEND_URL, json=packet_data)
            print(f"AI Response: {response.json()}")
        except Exception as e:
            print("❌ Could not connect to the Backend Server. Check the IP address.")

sniff(prn=packet_callback, count=10)
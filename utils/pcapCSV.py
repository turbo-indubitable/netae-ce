import csv
import sys
from scapy.all import rdpcap, RawPcapReader
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP, ICMP  # Fix: Import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6  # Fix: Import IPv6
from tqdm import tqdm
import os
from datetime import datetime

def parse_pcap(pcap_file, output_csv):
    file_name = os.path.basename(pcap_file)

    # Prepare CSV output file
    with open(output_csv, mode='w', newline='') as csvfile:
        fieldnames = [
            'file_name', 'timestamp',
            'src_ip', 'dst_ip', 'ip_version', 'src_port', 'dst_port',
            'protocol', 'tcp_flags', 'icmp_type', 'icmp_code',
            'packet_size', 'ttl'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        print("CSV File Output Initialized.")

        # Efficient total packet count computation using RawPcapReader
        total_packets = sum(1 for _ in RawPcapReader(pcap_file))
        print(f"Packet chunks measured. {total_packets} found to process. Starting...")

        # Load all packets in memory for normal processing using rdpcap
        packets = rdpcap(pcap_file)
        progress_bar = tqdm(total=total_packets, desc="Processing Packets", leave=True, dynamic_ncols=True, colour="white", file=sys.stdout)

        # Process packets
        for packet in packets:
            data = {
                'file_name': file_name,  # Add file name to each row
                'timestamp': packet.time if hasattr(packet, 'time') else None,  # Add packet timestamp
                'src_ip': None,
                'dst_ip': None,
                'ip_version': None,
                'src_port': None,
                'dst_port': None,
                'protocol': None,
                'tcp_flags': None,
                'icmp_type': None,
                'icmp_code': None,
                'packet_size': len(packet),
                'ttl': None
            }

            # Extract Layer 3 info (IP/IPv6)
            if IP in packet:  # equivalent to if packet.haslayer(IP)
                data['src_ip'] = packet[IP].src
                data['dst_ip'] = packet[IP].dst
                data['ttl'] = packet[IP].ttl
                data['protocol'] = packet[IP].proto
                data['ip_version'] = "4"  # Mark as IPv4

            elif IPv6 in packet:
                data['src_ip'] = packet[IPv6].src
                data['dst_ip'] = packet[IPv6].dst
                data['ttl'] = packet[IPv6].hlim
                data['protocol'] = packet[IPv6].nh
                data['ip_version'] = "6"  # Mark as IPv6

            # Extract Layer 4 info (TCP/UDP/ICMP)
            if TCP in packet:
                data['src_port'] = packet[TCP].sport
                data['dst_port'] = packet[TCP].dport
                data['tcp_flags'] = packet[TCP].flags

            elif UDP in packet:
                data['src_port'] = packet[UDP].sport
                data['dst_port'] = packet[UDP].dport

            elif ICMP in packet:
                data['icmp_type'] = packet[ICMP].type
                data['icmp_code'] = packet[ICMP].code

            # Write to CSV
            writer.writerow(data)
            progress_bar.update(1)

        progress_bar.close()
        print(f"Finished processing. Output saved to {output_csv}")


if __name__ == "__main__":
    today = datetime.now()
    pcap_file = '/dataprocessing/datafiles/pcap/CTU22__2017-05-02_kali-normal.pcap'
    output_csv = f"/home/phaze/PycharmProjects/NetworkModel/PCAPCSV_output/{today}.csv"

    if not os.path.exists(pcap_file):
        print("Error: PCAP file not found!")
    else:
        parse_pcap(pcap_file, output_csv)
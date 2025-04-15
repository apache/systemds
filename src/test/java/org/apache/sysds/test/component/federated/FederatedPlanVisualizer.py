import re
import networkx as nx
import matplotlib.pyplot as plt
import os
import glob

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
    HAS_PYGRAPHVIZ = True
except ImportError:
    HAS_PYGRAPHVIZ = False
    print("[주의] pygraphviz를 찾을 수 없습니다. 'pip install pygraphviz' 후 사용하세요.\n"
          "      설치가 안 된 경우 spring_layout 등 다른 레이아웃을 대체 사용합니다.")


def parse_line(line: str):
    # 원본 라인 출력
    print(f"원본 라인: {line}")
    
    # 빈 줄이거나 'Additional Cost:' 같은 정보 라인은 무시
    if not line or line.startswith("Additional Cost:"):
        return None
    
    # 1) 노드 ID 추출
    match_id = re.match(r'^\((R|\d+)\)', line)
    if not match_id:
        print(f"  > 노드 ID를 찾을 수 없음: {line}")
        return None
    node_id = match_id.group(1)
    print(f"  > 노드 ID: {node_id}")

    # 2) 노드 id 이후의 나머지 문자열
    after_id = line[match_id.end():].strip()
    print(f"  > ID 이후 문자열: {after_id}")

    # hop 이름(레이블): 첫 번째 "["가 나타나기 전까지의 문자열
    match_label = re.search(r'^(.*?)\s*\[', after_id)
    if match_label:
        operation = match_label.group(1).strip()
    else:
        operation = after_id.strip()
    print(f"  > Hop 이름/연산: {operation}")

    # 3) kind: 첫 번째 대괄호 안의 내용 (예: "FOUT" 또는 "LOUT")
    match_bracket = re.search(r'\[([^\]]+)\]', after_id)
    if match_bracket:
        kind = match_bracket.group(1).strip()
    else:
        kind = ""
    print(f"  > Kind: {kind}")

    # 4) total, self, weight: 중괄호 {} 안의 내용에서 추출
    total = ""
    self_cost = ""
    weight = ""
    match_curly = re.search(r'\{([^}]+)\}', line)
    if match_curly:
        curly_content = match_curly.group(1)
        m_total = re.search(r'Total:\s*([\d\.]+)', curly_content)
        m_self = re.search(r'Self:\s*([\d\.]+)', curly_content)
        m_weight = re.search(r'Weight:\s*([\d\.]+)', curly_content)
        if m_total:
            total = m_total.group(1)
        if m_self:
            self_cost = m_self.group(1)
        if m_weight:
            weight = m_weight.group(1)
    print(f"  > Total: {total}, Self: {self_cost}, Weight: {weight}")

    # 5) 참조 노드(child) 추출: kind 이후 첫 번째 괄호 안의 숫자들 (여러 개 가능)
    child_ids = []
    # 첫 번째 [ 다음에 나오는 괄호 찾기
    match_children = re.search(r'\[[^\]]+\]\s*\(([^)]+)\)', after_id)
    if match_children:
        children_str = match_children.group(1)
        print(f"  > 자식 노드 문자열: {children_str}")
        # 쉼표로 구분된 ID들 추출
        child_ids = [c.strip() for c in children_str.split(',') if c.strip()]
    print(f"  > 자식 노드 IDs: {child_ids}")
    
    # 6) 엣지 세부 정보: [Edges]{...}에서 추출
    edge_details = {}
    match_edges = re.search(r'\[Edges\]\{(.*?)(?:\}|$)', line)
    if match_edges:
        edges_str = match_edges.group(1)
        print(f"  > [Edges] 내용: {edges_str}")
        
        # 각 엣지 정보를 괄호 단위로 분리
        edge_items = re.findall(r'\(ID:[^)]+\)', edges_str)
        
        for item in edge_items:
            print(f"  > 파싱할 부분: '{item}'")
            
            # 엣지 정보 파싱: (ID:51, X, C:401810.0, F:0.0, FW:500.0)
            id_match = re.search(r'ID:(\d+)', item)
            xo_match = re.search(r',\s*([XO])', item)
            cumulative_match = re.search(r'C:([\d\.]+)', item)
            forward_match = re.search(r'F:([\d\.]+)', item)
            weight_match = re.search(r'FW:([\d\.]+)', item)
            
            if id_match:
                source_id = id_match.group(1)
                is_forwarding = xo_match and xo_match.group(1) == 'O'
                cumulative_cost = cumulative_match.group(1) if cumulative_match else None
                forward_cost = forward_match.group(1) if forward_match else "0.0"
                forward_weight = weight_match.group(1) if weight_match else "1.0"
                
                print(f"  > 엣지 상세 정보 파싱: source={source_id}, forwarding={'O' if is_forwarding else 'X'}, cumulative={cumulative_cost}, cost={forward_cost}, weight={forward_weight}")
                
                edge_details[source_id] = {
                    'is_forwarding': is_forwarding,
                    'cumulative_cost': cumulative_cost,
                    'forward_cost': forward_cost,
                    'forward_weight': forward_weight
                }

    print(f"  > 엣지 상세 정보: {edge_details}")
    print("-------------------------------------")

    return {
        'node_id': node_id,
        'operation': operation,
        'kind': kind,
        'total': total,
        'self_cost': self_cost,
        'weight': weight,
        'child_ids': child_ids,
        'edge_details': edge_details
    }


def build_dag_from_file(filename: str):
    G = nx.DiGraph()
    print(f"\n[INFO] 파일 '{filename}'에서 그래프를 구성합니다.")
    
    line_count = 0
    parsed_count = 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue

            info = parse_line(line)
            if not info:
                continue
                
            parsed_count += 1
            node_id = info['node_id']
            operation = info['operation']
            kind = info['kind']
            total = info['total']
            self_cost = info['self_cost']
            weight = info['weight']
            child_ids = info['child_ids']
            edge_details = info['edge_details']

            print(f"노드 추가: {node_id}, 레이블: {operation}, 종류: {kind}")
            G.add_node(node_id, label=operation, kind=kind, total=total, self_cost=self_cost, weight=weight)

            # 1. 먼저 () 안에 있는 자식 ID로 기본 엣지 생성
            for child_id in child_ids:
                # 자식 노드가 아직 없으면 생성
                if child_id not in G:
                    print(f"  > 없는 자식 노드 생성: {child_id}")
                    G.add_node(child_id, label=child_id, kind="", total="", self_cost="", weight="")
                
                # 자식 노드에서 현재 노드로 가는 엣지 추가 (자식 -> 부모)
                # 기본값으로 설정 (미발견 엣지는 -1로 표시)
                print(f"  > 기본 엣지 추가: {child_id} -> {node_id} (미발견 엣지)")
                G.add_edge(child_id, node_id, 
                          is_forwarding=False,
                          forward_cost="-1",  # 미발견 엣지는 -1로 표시
                          forward_weight="-1",  # 미발견 엣지는 -1로 표시
                          is_discovered=False)  # 추가 플래그
            
            # 2. [Edges] 정보로 엣지 속성 업데이트
            for source_id, edge_data in edge_details.items():
                # 소스 노드가 없으면 생성
                if source_id not in G:
                    print(f"  > 없는 소스 노드 생성: {source_id}")
                    G.add_node(source_id, label=source_id, kind="", total="", self_cost="", weight="")
                
                # 엣지가 아직 없으면 생성, 있으면 속성만 업데이트
                if not G.has_edge(source_id, node_id):
                    # 엣지 속성 설정
                    edge_attrs = {
                        'is_forwarding': edge_data['is_forwarding'],
                        'forward_cost': edge_data['forward_cost'],
                        'forward_weight': edge_data['forward_weight'],
                        'is_discovered': True  # [Edges]에서 발견된 엣지
                    }
                    
                    # 누적 비용이 있으면 추가
                    if 'cumulative_cost' in edge_data and edge_data['cumulative_cost'] is not None:
                        edge_attrs['cumulative_cost'] = edge_data['cumulative_cost']
                        
                    print(f"  > 엣지 추가: {source_id} -> {node_id}, Forwarding: {edge_data['is_forwarding']}, Cost: {edge_data['forward_cost']}, Weight: {edge_data['forward_weight']}, Cumulative: {edge_data['cumulative_cost']}")
                    G.add_edge(source_id, node_id, **edge_attrs)
                else:
                    print(f"  > 엣지 속성 업데이트: {source_id} -> {node_id}, Forwarding: {edge_data['is_forwarding']}, Cost: {edge_data['forward_cost']}, Weight: {edge_data['forward_weight']}, Cumulative: {edge_data['cumulative_cost']}")
                    G[source_id][node_id]['is_forwarding'] = edge_data['is_forwarding']
                    G[source_id][node_id]['forward_cost'] = edge_data['forward_cost']
                    G[source_id][node_id]['forward_weight'] = edge_data['forward_weight']
                    G[source_id][node_id]['is_discovered'] = True  # Edges에서 발견된 엣지
                    
                    # 누적 비용이 있으면 추가
                    if 'cumulative_cost' in edge_data and edge_data['cumulative_cost'] is not None:
                        G[source_id][node_id]['cumulative_cost'] = edge_data['cumulative_cost']

    print(f"\n[INFO] 총 {line_count}줄 중 {parsed_count}개의 노드를 파싱했습니다.")
    print(f"[INFO] 그래프 정보: 노드 {len(G.nodes())}개, 엣지 {len(G.edges())}개\n")
    
    print("--- 노드 정보 ---")
    for node, data in G.nodes(data=True):
        print(f"노드 {node}: {data}")
    
    print("\n--- 엣지 정보 ---")
    for u, v, data in G.edges(data=True):
        print(f"엣지 {u} -> {v}: {data}")
    
    return G


def get_unique_filename(base_filename: str) -> str:
    """기존 파일이 있으면 increment하여 새로운 파일명을 생성"""
    if not os.path.exists(base_filename):
        return base_filename
    
    name, ext = os.path.splitext(base_filename)
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1


def visualize_plan(filename: str, output_dir: str = "visualization_output"):
    print(f"[INFO] 파일 '{filename}'을 시각화합니다.")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    G = build_dag_from_file(filename)
    print("Nodes:", G.nodes(data=True))
    print("Edges:", list(G.edges(data=True)))

    if HAS_PYGRAPHVIZ:
        # 노드 간격을 더 크게 설정 (nodesep: 노드 간 수평 간격, ranksep: 레벨 간 수직 간격)
        pos = graphviz_layout(G, prog='dot', args='-Grankdir=BT -Gnodesep=3 -Granksep=3')
    else:
        # spring_layout의 경우 k 값을 크게 하여 노드 간 간격 확보
        pos = nx.spring_layout(G, seed=42, k=2.0)

    # 노드 개수에 따라 전체 그래프의 크기를 동적으로 조절
    node_count = len(G.nodes())
    fig_width = 15 + node_count / 8.0  # 가로 크기 증가
    fig_height = 10 + node_count / 8.0  # 세로 크기 증가
    plt.figure(figsize=(fig_width, fig_height), facecolor='white', dpi=300)
    ax = plt.gca()
    ax.set_facecolor('white')

    # 노드 레이블 설정 (형식: id: hop 이름 \n Total \n Self)
    labels = {}
    for n in G.nodes():
        # 기본 정보
        node_id = n
        label = G.nodes[n].get('label', n)
        total_cost = G.nodes[n].get('total', '')
        self_cost = G.nodes[n].get('self_cost', '')
        weight = G.nodes[n].get('weight', '')
        
        # 자식 엣지를 순회하여 누적 비용과 포워딩 비용 합계 계산
        child_cumulated_cost_sum = 0.0
        child_forward_cost_sum = 0.0
        
        print(f"\n[DEBUG] 노드 {node_id}의 child 비용 계산:")
        
        # 1. 이 노드로 들어오는 모든 엣지 (자식 노드들) 찾기
        child_nodes = []
        for child, _, _ in G.in_edges(n, data=True):
            child_nodes.append(child)
        
        print(f"  자식 노드들: {child_nodes}")
        
        # 2. 각 자식 노드의 cumulative_cost와 forward_cost 합산
        for child_node in child_nodes:
            # 자식 노드의 총 비용 (Total)
            child_total = G.nodes[child_node].get('total', '0.0')
            try:
                child_total_float = float(child_total)
                print(f"  자식 노드 {child_node}의 Total 비용: {child_total_float}")
                child_cumulated_cost_sum += child_total_float
            except (ValueError, TypeError):
                print(f"  자식 노드 {child_node}의 Total 비용 변환 실패: {child_total}")
            
            # 자식 노드의 포워딩 비용 계산
            # 자식 노드에서 나가는 엣지들의 forward_cost 합산
            child_forward_sum = 0.0
            for _, grandchild, data in G.out_edges(child_node, data=True):
                if 'forward_cost' in data and data['forward_cost'] is not None:
                    try:
                        if data['forward_cost'] != '-1':  # 미발견 엣지가 아닌 경우에만
                            fwd_cost = float(data['forward_cost'])
                            child_forward_sum += fwd_cost
                            print(f"    자식 노드 {child_node}의 forward_cost: {fwd_cost}")
                    except ValueError:
                        print(f"    자식 노드 {child_node}의 forward_cost 변환 실패: {data['forward_cost']}")
            
            child_forward_cost_sum += child_forward_sum
        
        # 레이블 첫 줄: 노드 ID, 연산, 총 비용, 가중치
        first_line = f"{node_id}: {label}"
        if total_cost:
            # 정수 부분만 출력
            try:
                first_line += f"\nC: {int(float(total_cost))}"
            except (ValueError, TypeError):
                first_line += f"\nC: {total_cost}"
        if weight:
            # 정수 부분만 출력
            try:
                first_line += f", W: {int(float(weight))}"
            except (ValueError, TypeError):
                first_line += f", W: {weight}"
        
        # 레이블 두 번째 줄: Self Cost, 자식 누적 비용 합, 자식 포워딩 비용 합을 슬래시(/)로 구분
        # 정수 부분만 출력
        try:
            self_cost_int = int(float(self_cost)) if self_cost else 0
        except (ValueError, TypeError):
            self_cost_int = 0
        
        child_cumulated_cost_int = int(child_cumulated_cost_sum)
        child_forward_cost_int = int(child_forward_cost_sum)
        
        print(f"  최종 비용 합계: Self={self_cost_int}, Child Total={child_cumulated_cost_int}, Child Fwd={child_forward_cost_int}")
        second_line = f"({self_cost_int}/{child_cumulated_cost_int}/{child_forward_cost_int})"
        
        # 최종 레이블
        labels[n] = f"{first_line}\n{second_line}"

    # 노드별 색상 결정 (kind에 따라)
    def get_color(n):
        k = G.nodes[n].get('kind', '').lower()
        if k == 'fout':
            return 'tomato'
        elif k == 'lout':
            return 'dodgerblue'
        elif k == 'nref':
            return 'mediumpurple'
        else:
            return 'mediumseagreen'

    # 노드 모양 결정 (node의 label에 해당 문자열이 포함되는지 검사):
    # 'twrite'가 포함되면 세모(삼각형, marker '^')
    # 'tread'가 포함되면 네모(정사각형, marker 's')
    # 그 외는 원(circle, marker 'o')
    triangle_nodes = [n for n in G.nodes() if 'twrite' in G.nodes[n].get('label', '').lower()]
    square_nodes = [n for n in G.nodes() if 'tread' in G.nodes[n].get('label', '').lower()]
    other_nodes = [n for n in G.nodes() 
                   if 'twrite' not in G.nodes[n].get('label', '').lower() and
                      'tread' not in G.nodes[n].get('label', '').lower()]

    triangle_colors = [get_color(n) for n in triangle_nodes]
    square_colors = [get_color(n) for n in square_nodes]
    other_colors = [get_color(n) for n in other_nodes]

    # 노드 크기 증가
    node_size = 1200

    # 각각의 노드 그룹을 별도로 그리기
    node_collection_triangle = nx.draw_networkx_nodes(G, pos, nodelist=triangle_nodes, node_size=node_size, 
                                                      node_color=triangle_colors, node_shape='^', ax=ax)
    node_collection_square = nx.draw_networkx_nodes(G, pos, nodelist=square_nodes, node_size=node_size, 
                                                    node_color=square_colors, node_shape='s', ax=ax)
    node_collection_other = nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_size=node_size, 
                                                   node_color=other_colors, node_shape='o', ax=ax)

    # zorder 조절 (노드:1, 에지:2, 레이블:3)
    node_collection_triangle.set_zorder(1)
    node_collection_square.set_zorder(1)
    node_collection_other.set_zorder(1)

    # 엣지를 forwarding 발생 여부와 ROOT 노드 연결 여부에 따라 다른 색상으로 그리기
    
    # 1. 일반 엣지 (ROOT 노드와 무관한 엣지)
    normal_forwarding_edges = [(u, v) for u, v, d in G.edges(data=True) 
                              if 'is_discovered' in d and d['is_discovered'] 
                              and 'is_forwarding' in d and d['is_forwarding']
                              and v != 'R' and u != 'R']
    
    normal_non_forwarding_edges = [(u, v) for u, v, d in G.edges(data=True) 
                                  if 'is_discovered' in d and d['is_discovered'] 
                                  and 'is_forwarding' in d and not d['is_forwarding']
                                  and v != 'R' and u != 'R']
    
    # 2. ROOT 노드에 연결된 모든 엣지 (발견/미발견 모두 포함하여 검정색으로 표시)
    root_edges = [(u, v) for u, v, d in G.edges(data=True) 
                 if v == 'R' or u == 'R']
    
    # 3. 미발견 엣지 (ROOT 노드에 연결된 것은 제외)
    undiscovered_edges = [(u, v) for u, v, d in G.edges(data=True) 
                         if ('is_discovered' not in d or not d['is_discovered'])
                         and v != 'R' and u != 'R']
    
    print(f"\n[DEBUG] 일반 Forwarding 발생 엣지: {normal_forwarding_edges}")
    print(f"[DEBUG] 일반 Forwarding 미발생 엣지: {normal_non_forwarding_edges}")
    print(f"[DEBUG] ROOT 연결 엣지: {root_edges}")
    print(f"[DEBUG] 미발견 엣지: {undiscovered_edges}")
    
    # 일반 forwarding 발생 엣지: 빨간색
    normal_forwarding_collection = nx.draw_networkx_edges(G, pos, edgelist=normal_forwarding_edges, 
                          arrows=True, arrowstyle='->', 
                          edge_color='red', width=2.0, ax=ax)
    
    # 일반 forwarding 미발생 엣지: 검은색
    normal_non_forwarding_collection = nx.draw_networkx_edges(G, pos, edgelist=normal_non_forwarding_edges, 
                          arrows=True, arrowstyle='->', 
                          edge_color='black', width=1.0, ax=ax)
    
    # ROOT 노드 연결 모든 엣지: 검은색
    root_edges_collection = nx.draw_networkx_edges(G, pos, edgelist=root_edges, 
                          arrows=True, arrowstyle='->', 
                          edge_color='black', width=1.0, ax=ax)
    
    # 미발견 엣지: 보라색 굵은 선
    undiscovered_collection = nx.draw_networkx_edges(G, pos, edgelist=undiscovered_edges, 
                                                       arrows=True, arrowstyle='->', 
                                                       edge_color='purple', width=2.5, alpha=0.7, ax=ax)
    
    # z-order 설정을 위한 도우미 함수
    def set_zorder_for_collection(collection, z=2):
        if isinstance(collection, list):
            for ec in collection:
                ec.set_zorder(z)
        elif collection is not None:
            collection.set_zorder(z)
    
    # 모든 엣지 컬렉션에 z-order 설정
    set_zorder_for_collection(normal_forwarding_collection)
    set_zorder_for_collection(normal_non_forwarding_collection)
    set_zorder_for_collection(root_edges_collection)
    set_zorder_for_collection(undiscovered_collection)

    # 엣지 레이블 추가 (forwarding cost와 weight 정보) - 배경을 완전히 투명하게 설정
    edge_labels = {}
    
    # 발견된 엣지는 C/W/CC 형식으로 표시 (ROOT 노드 연결 제외)
    for u, v, d in G.edges(data=True):
        # ROOT 노드에 연결된 엣지는 레이블 표시 안함
        if v == 'R' or u == 'R':
            continue
            
        # 발견된 엣지는 정보 표시
        if 'is_discovered' in d and d['is_discovered'] and 'forward_cost' in d and 'forward_weight' in d:
            label_parts = []
            
            # 포워딩 비용 (정수 부분만)
            try:
                forward_cost_int = int(float(d['forward_cost']))
                label_parts.append(f"C:{forward_cost_int}")
            except ValueError:
                label_parts.append(f"C:{d['forward_cost']}")
            
            # 가중치 (정수 부분만)
            try:
                forward_weight_int = int(float(d['forward_weight']))
                label_parts.append(f"W:{forward_weight_int}")
            except ValueError:
                label_parts.append(f"W:{d['forward_weight']}")
            
            # # 누적 비용이 있으면 추가 (정수 부분만)
            # if 'cumulative_cost' in d and d['cumulative_cost'] is not None:
            #     try:
            #         cumulative_cost_int = int(float(d['cumulative_cost']))
            #         label_parts.append(f"C:{cumulative_cost_int}")
            #     except ValueError:
            #         label_parts.append(f"C:{d['cumulative_cost']}")
            
            edge_labels[(u, v)] = "\n".join(label_parts)
        # 미발견 엣지는 "Undiscovered"로 표시
        elif ('is_discovered' not in d or not d['is_discovered']) and 'forward_cost' in d and 'forward_weight' in d:
            edge_labels[(u, v)] = "Undiscovered"

    # 엣지 레이블 추가 - 배경을 완전히 투명하게 설정
    edge_label_dict = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                                 font_size=7, font_color='darkblue',
                                                 bbox=dict(boxstyle="round", fc="w", ec="none", alpha=0),
                                                 ax=ax)
    
    # 레이블 배경을 직접 투명하게 설정
    for key, text in edge_label_dict.items():
        text.set_bbox(dict(boxstyle="round", fc="none", ec="none", alpha=0))

    # 노드 레이블 - 배경을 완전히 투명하게 설정
    label_dict = nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, 
                                       bbox=dict(boxstyle="round", fc="w", ec="none", alpha=0),
                                       ax=ax)
    
    # 노드 레이블의 배경도 직접 투명하게 설정
    for text in label_dict.values():
        text.set_zorder(3)
        text.set_bbox(dict(boxstyle="round", fc="none", ec="none", alpha=0))

    # 원하는 타이틀 설정
    plt.title("Program Level Federated Plan", fontsize=16, fontweight="bold")

    # 노드 유형 범례 (좌측 상단)
    plt.scatter(0.05, 0.95, color='dodgerblue', s=150, transform=ax.transAxes)
    plt.scatter(0.18, 0.95, color='tomato', s=150, transform=ax.transAxes)
    plt.scatter(0.31, 0.95, color='mediumpurple', s=150, transform=ax.transAxes)

    plt.text(0.08, 0.95, "LOUT", fontsize=10, va='center', transform=ax.transAxes)
    plt.text(0.21, 0.95, "FOUT", fontsize=10, va='center', transform=ax.transAxes)
    plt.text(0.34, 0.95, "NREF", fontsize=10, va='center', transform=ax.transAxes)
    
    # Edge 관련 범례 (우측 상단)
    legend_x = 0.98  # 우측 상단 x 좌표
    legend_y = 0.98  # 우측 상단 y 좌표
    legend_spacing = 0.05  # 각 항목 간 간격

    # 레이블 범례 (텍스트만)
    plt.text(legend_x, legend_y, "[Node LABEL]\nhopID: hopNam\nC: Total Cost, W: Weight\n(Self / Child Cum. Cost / Child Fwd. Cost)", 
             fontsize=12, ha='right', va='top', transform=ax.transAxes)

    # # 엣지 유형 범례
    # y_offset = legend_y - 0.3  # 엣지 범례 시작 y 위치
    
    # # 엣지 유형 제목
    # plt.text(legend_x, y_offset, "Edge Types:", 
    #          fontsize=12, ha='right', va='center', transform=ax.transAxes)
    # y_offset -= legend_spacing
    
    # # Forwarding 엣지
    # plt.plot([legend_x-0.13, legend_x-0.08], [y_offset, y_offset], 
    #          color='red', linewidth=2, transform=ax.transAxes)
    # plt.text(legend_x, y_offset, "Forwarding Cost (O)", 
    #          fontsize=10, ha='right', va='center', transform=ax.transAxes)
    # y_offset -= legend_spacing
    
    # # No Forwarding 엣지
    # plt.plot([legend_x-0.13, legend_x-0.08], [y_offset, y_offset], 
    #          color='black', linewidth=1, transform=ax.transAxes)
    # plt.text(legend_x, y_offset, "No Forwarding Cost", 
    #          fontsize=10, ha='right', va='center', transform=ax.transAxes)
    # y_offset -= legend_spacing
    
    # # Undiscovered 엣지
    # plt.plot([legend_x-0.13, legend_x-0.08], [y_offset, y_offset], 
    #          color='purple', linewidth=2.5, alpha=0.7, transform=ax.transAxes)
    # plt.text(legend_x, y_offset, "Undiscovered", 
    #          fontsize=10, ha='right', va='center', transform=ax.transAxes)

    plt.axis("off")

    # 입력 파일 이름을 기반으로 출력 파일 이름 생성
    input_filename = os.path.basename(filename)
    base_output_filename = os.path.splitext(input_filename)[0] + ".png"
    output_filename = os.path.join(output_dir, base_output_filename)
    
    # 중복 파일명 처리
    output_filename = get_unique_filename(output_filename)
    
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"[INFO] 시각화 결과가 '{output_filename}'에 저장되었습니다.")
    plt.close()


def main():
    import sys
    print("사용법: python FederatedPlanVisualizer.py <trace_file>")
    if len(sys.argv) != 2:
        print("사용법: python FederatedPlanVisualizer.py <trace_file>")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    if not os.path.exists(trace_file):
        print(f"[오류] 파일 '{trace_file}'을 찾을 수 없습니다.")
        sys.exit(1)
    
    visualize_plan(trace_file)


if __name__ == '__main__':
    main()

#pragma once

#ifdef USE_FIBONACCI_HEAP
#include <boost/heap/fibonacci_heap.hpp>
#endif

#include <boost/heap/d_ary_heap.hpp>
#include <unordered_map>
#include <unordered_set>

#include "neighbor.hpp"
#include "planresult.hpp"

namespace libMultiRobotPlanning {

/*!
  \example a_star.cpp Simple example using a 2D grid world and
  up/down/left/right
  actions
*/

/*! \brief A* Algorithm to find the shortest path

This class implements the A* algorithm. A* is an informed search algorithm
that finds the shortest path for a given map. It can use a heuristic that
needsto be admissible.

This class can either use a fibonacci heap, or a d-ary heap. The latter is the
default. Define "USE_FIBONACCI_HEAP" to use the fibonacci heap instead.

\tparam State Custom state for the search. Needs to be copy'able
\tparam Action Custom action for the search. Needs to be copy'able
\tparam Cost Custom Cost type (integer or floating point types)
\tparam Environment This class needs to provide the custom A* logic. In
    particular, it needs to support the following functions:
  - `Cost admissibleHeuristic(const State& s)`\n
    This function can return 0 if no suitable heuristic is available.

  - `bool isSolution(const State& s)`\n
    Return true if the given state is a goal state.

  - `void getNeighbors(const State& s, std::vector<Neighbor<State, Action,
   int> >& neighbors)`\n
    Fill the list of neighboring state for the given state s.

  - `void onExpandNode(const State& s, int fScore, int gScore)`\n
    This function is called on every expansion and can be used for statistical
purposes.

  - `void onDiscover(const State& s, int fScore, int gScore)`\n
    This function is called on every node discovery and can be used for
   statistical purposes.

    \tparam StateHasher A class to convert a state to a hash value. Default:
   std::hash<State>
*/
template <typename State, typename Action, typename Cost, typename Environment,
          typename StateHasher = std::hash<State> >
class AStar {
 public:
  AStar(Environment& environment) : m_env(environment) {}

  bool search(const State& startState,
              PlanResult<State, Action, Cost>& solution, Cost initialCost = 0) {
    solution.states.clear();
    solution.states.push_back(std::make_pair<>(startState, 0));
    solution.actions.clear();
    solution.cost = 0;

    openSet_t openSet; //--由于最后有个fibheap，猜测这是斐波那契堆
    //-- std::unordered_map
    //--内部无序，键值对，stateToHeap, 容器元素是pair类型，pair.first = key, pair.second = value
    //--通过key查询value，复杂度O(1)
    std::unordered_map<State, fibHeapHandle_t, StateHasher> stateToHeap; 
    std::unordered_set<State, StateHasher> closedSet;
    std::unordered_map<State, std::tuple<State, Action, Cost, Cost>,
                       StateHasher>
        cameFrom;
    //--Node中包含 State,H(启发式的值)
    //--把起点放到open_list中
    auto handle = openSet.push(
        Node(startState, m_env.admissibleHeuristic(startState), initialCost));
    stateToHeap.insert(std::make_pair<>(startState, handle));
    (*handle).handle = handle;

    std::vector<Neighbor<State, Action, Cost> > neighbors;
    neighbors.reserve(10); //--指定vector的容量是10,并且之后push_back的数量超过指定的值后，动态分配是其倍数20

    while (!openSet.empty()) {
      //--弹出代价最小的，用的最小堆
      Node current = openSet.top();
      //--扩展节点
      m_env.onExpandNode(current.state, current.fScore, current.gScore);
      //--检查终点是否在open_list中，如果在,就对之前的solution进行clear，然后填补
      if (m_env.isSolution(current.state)) {
        solution.states.clear();
        solution.actions.clear();
        auto iter = cameFrom.find(current.state);
        while (iter != cameFrom.end()) {
          solution.states.push_back(
              std::make_pair<>(iter->first, std::get<3>(iter->second))); //-- 状态和G值
          solution.actions.push_back(std::make_pair<>(
              std::get<1>(iter->second), std::get<2>(iter->second)));//-- Action和F值
          iter = cameFrom.find(std::get<0>(iter->second)); //--find状态(应该是父节点)
        }
        solution.states.push_back(std::make_pair<>(startState, initialCost));
        std::reverse(solution.states.begin(), solution.states.end());
        std::reverse(solution.actions.begin(), solution.actions.end());
        solution.cost = current.gScore;
        solution.fmin = current.fScore;

        return true;
      }
      //--如果不是终点，那就把当前节点从open_list中除去pop，然后在close_list中加入
      openSet.pop();
      stateToHeap.erase(current.state); //--在stateToHeap中删除当前节点的状态
      closedSet.insert(current.state);

      // traverse neighbors
      neighbors.clear();
      m_env.getNeighbors(current.state, neighbors); //--会检查约束
      for (const Neighbor<State, Action, Cost>& neighbor : neighbors) {
        if (closedSet.find(neighbor.state) == closedSet.end()) { //--检查不在close_list中
          Cost tentative_gScore = current.gScore + neighbor.cost; //--当前G值
          auto iter = stateToHeap.find(neighbor.state);
          if (iter == stateToHeap.end()) {  // Discover a new node //--如果扩展的节点状态不在open中
            Cost fScore =
                tentative_gScore + m_env.admissibleHeuristic(neighbor.state); //--F=G+曼哈顿距离
            auto handle =
                openSet.push(Node(neighbor.state, fScore, tentative_gScore));
            (*handle).handle = handle; //--应该就是指针那种类似的  当前handle，指向的是neighbor
            stateToHeap.insert(std::make_pair<>(neighbor.state, handle));
            m_env.onDiscover(neighbor.state, fScore, tentative_gScore); //--函数中好像没有代码
            // std::cout << "  this is a new node " << fScore << "," <<
            // tentative_gScore << std::endl;
          } else {
            auto handle = iter->second; //--若在open_list，就看看是否需要更新其父节点。iter是stateToHeap 应该是哈希值
            //--std::unordered_map<State, fibHeapHandle_t, StateHasher> stateToHeap; 
            // std::cout << "  this is an old node: " << tentative_gScore << ","
            // << (*handle).gScore << std::endl;
            // We found this node before with a better path
            if (tentative_gScore >= (*handle).gScore) { //--如果当前状态的哈希对应的状态的G大于，就不用做什么操作,直接到下一个neigh
                                                        //--如果小于，就需要把*handle对应F和G值进行修改
              continue;
            }

            // update f and gScore
            Cost delta = (*handle).gScore - tentative_gScore;
            (*handle).gScore = tentative_gScore;
            (*handle).fScore -= delta;
            openSet.increase(handle); //--修改G和F值后，在open_set中重新排序，数据结构必须配置为可变的
            m_env.onDiscover(neighbor.state, (*handle).fScore,
                             (*handle).gScore);
          }

          // Best path for this node so far
          // TODO: this is not the best way to update "cameFrom", but otherwise
          // default c'tors of State and Action are required
          cameFrom.erase(neighbor.state);
          cameFrom.insert(std::make_pair<>(
              neighbor.state,
              std::make_tuple<>(current.state, neighbor.action, neighbor.cost,
                                tentative_gScore))); //--tuple第一个是父节点
        }
      }
    }

    return false;
  }

 private:
  struct Node {
    Node(const State& state, Cost fScore, Cost gScore)
        : state(state), fScore(fScore), gScore(gScore) {}

    bool operator<(const Node& other) const {
      // Sort order
      // 1. lowest fScore
      // 2. highest gScore

      // Our heap is a maximum heap, so we invert the comperator function here
      if (fScore != other.fScore) {
        return fScore > other.fScore;
      } else {
        return gScore < other.gScore;
      }
    }

    friend std::ostream& operator<<(std::ostream& os, const Node& node) {
      os << "state: " << node.state << " fScore: " << node.fScore
         << " gScore: " << node.gScore;
      return os;
    }

    State state;

    Cost fScore;
    Cost gScore;

#ifdef USE_FIBONACCI_HEAP
    typename boost::heap::fibonacci_heap<Node>::handle_type handle;
#else
    typename boost::heap::d_ary_heap<Node, boost::heap::arity<2>,
                                     boost::heap::mutable_<true> >::handle_type
        handle;
#endif
  };

#ifdef USE_FIBONACCI_HEAP
  typedef typename boost::heap::fibonacci_heap<Node> openSet_t;
  typedef typename openSet_t::handle_type fibHeapHandle_t;
#else
  typedef typename boost::heap::d_ary_heap<Node, boost::heap::arity<2>,
                                           boost::heap::mutable_<true> >
      openSet_t;
  typedef typename openSet_t::handle_type fibHeapHandle_t;
#endif

 private:
  Environment& m_env;
};

}  // namespace libMultiRobotPlanning

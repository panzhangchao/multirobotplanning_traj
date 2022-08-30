#!/usr/bin/python
import rospy
import copy
import tf
import numpy as np
from scipy import spatial
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from pyomo.environ import *
from pyomo.dae import *
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
# % matplotlib inline
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

wps = np.loadtxt("wps.txt")
x = wps[:,0]
y = wps[:,1]
t = np.linspace(0, 1, num=len(x))
f1 = interp1d(t,x,kind='cubic')
f2 = interp1d(t,y,kind='cubic')
newt = np.linspace(0,1,100)
nwps = np.zeros((100, 2))
nwps[:,0] = f1(newt)
nwps[:,1] = f2(newt)
wpstree = spatial.KDTree(nwps)

def getcwps(rp):
    _, nindex = wpstree.query(rp)
    cwps = np.zeros((5,2))
    for i in range(5):
        cwps[i] = nwps[(nindex+i)%len(nwps)]
        
#     if (nindex + 5) >= 100:
#         cwps[0:100-nindex-1] = nwps[nindex:-1]
#         cwps[100-nindex-1:-1] = nwps[0:nindex+5-100]        
#     else:
#         cwps = nwps[nindex:nindex+5]
    return cwps    

def cubic_fun(coeffs, x):
    return coeffs[0]*x**3+coeffs[1]*x**2+coeffs[2]*x+coeffs[0]    
        
def plot_durations(cwps, prex, prey):
    plt.figure(2)
    plt.clf()
    plt.plot(cwps[:,0],cwps[:,1])
    plt.plot(prex, prey)
    plt.scatter(x, y)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
 
        
        
N = 19 # forward predict steps
ns = 5  # state numbers / here: 1: x, 2: y, 3: psi, 4: cte, 5: epsi
na = 2  # actuator numbers /here: 1: steering angle, 2: omega


class MPC(object):
    def __init__(self):
        m = ConcreteModel()
        m.sk = RangeSet(0, N-1)
        m.uk = RangeSet(0, N-2)
        m.uk1 = RangeSet(0, N-3)
        
        m.wg       = Param(RangeSet(0, 3), initialize={0:1., 1:10., 2:100., 3:1000}, mutable=True) 
        m.dt       = Param(initialize=0.1, mutable=True)
        m.ref_v    = Param(initialize=0.5, mutable=True)
        m.ref_cte  = Param(initialize=0.0, mutable=True)
        m.ref_epsi = Param(initialize=0.0, mutable=True)
        m.s0       = Param(RangeSet(0, ns-1), initialize={0:0., 1:0., 2:0., 3:0., 4:0.}, mutable=True)
        m.coeffs   = Param(RangeSet(0, 3), 
                          initialize={0:-0.000458316, 1:0.00734257, 2:0.0538795, 3:0.080728}, mutable=True)
        
        
        m.s      = Var(RangeSet(0, ns-1), m.sk)
        m.f      = Var(m.sk)
        m.psides = Var(m.sk)
        m.uv     = Var(m.uk, bounds=(-0.01, 2.0))
        m.uw     = Var(m.uk, bounds=(-1.5, 1.5))
        
        # 0: x, 1: y, 2: psi, 3: cte, 4: epsi
        m.s0_update      = Constraint(RangeSet(0, ns-1), rule = lambda m, i: m.s[i,0] == m.s0[i])
        m.x_update       = Constraint(m.sk, rule=lambda m, k: 
                                      m.s[0,k+1]==m.s[0,k]+m.uv[k]*cos(m.s[2,k])*m.dt 
                                      if k<N-1 else Constraint.Skip)
        m.y_update       = Constraint(m.sk, rule=lambda m, k: 
                                      m.s[1,k+1]==m.s[1,k]+m.uv[k]*sin(m.s[2,k])*m.dt 
                                      if k<N-1 else Constraint.Skip)
        m.psi_update     = Constraint(m.sk, rule=lambda m, k: 
                                       m.s[2,k+1]==m.s[2,k]+ m.uw[k]*m.dt 
                                       if k<N-1 else Constraint.Skip)     
        m.f_update      = Constraint(m.sk, rule=lambda m, k: 
                                       m.f[k]==m.coeffs[0]*m.s[0,k]**3+m.coeffs[1]*m.s[0,k]**2+
                                       m.coeffs[2]*m.s[0,k]+m.coeffs[3])
        m.psides_update = Constraint(m.sk, rule=lambda m, k: 
                                           m.psides[k]==atan(3*m.coeffs[0]*m.s[0,k]**2
                                                              +2*m.coeffs[1]*m.s[0,k]+m.coeffs[2]))
        m.cte_update     = Constraint(m.sk, rule=lambda m, k: 
                                        m.s[3,k+1]==(m.f[k]-m.s[1,k]+m.uv[k]*sin(m.s[2,k])*m.dt) 
                                       if k<N-1 else Constraint.Skip)

        m.epsi_update    = Constraint(m.sk, rule=lambda m, k: 
                                   m.s[4, k+1]==m.psides[k]-m.s[2,k]+m.uw[k]*m.dt 
                                        if k<N-1 else Constraint.Skip)  
        
        m.cteobj  = m.wg[3]*sum((m.s[3,k]-m.ref_cte)**2 for k in m.sk)
        m.epsiobj = m.wg[3]*sum((m.s[4,k]-m.ref_epsi)**2 for k in m.sk)
        m.vobj    = m.wg[2]*sum((m.uv[k]-0.5)**2 for k in m.uk)
        m.uvobj   = m.wg[1]*sum(m.uv[k]**2 for k in m.uk)
        m.uwobj   = m.wg[1]*sum(m.uw[k]**2 for k in m.uk)
        m.sudobj  = m.wg[0]*sum((m.uv[k+1]-m.uv[k])**2 for k in m.uk1)
        m.suaobj  = m.wg[0]*sum((m.uw[k+1]-m.uw[k])**2 for k in m.uk1) 
        m.obj = Objective(expr = m.cteobj+m.epsiobj+m.vobj+m.uvobj+m.uwobj+m.sudobj+m.suaobj, sense=minimize)
        
        self.iN = m#.create_instance()
        
    def Solve(self, state, coeffs):        
        self.iN.s0.reconstruct({0:state[0], 1: state[1], 2:state[2], 3:state[3], 4:state[4]})
        self.iN.coeffs.reconstruct({0:coeffs[0], 1:coeffs[1], 2:coeffs[2], 3:coeffs[3]})
        self.iN.f_update.reconstruct()
        self.iN.s0_update.reconstruct()
        self.iN.psides_update.reconstruct()
        SolverFactory('ipopt').solve(self.iN)
        x_pred_vals = [self.iN.s[0,k]() for k in self.iN.sk]
        y_pred_vals = [self.iN.s[1,k]() for k in self.iN.sk]
        pre_path = np.zeros((N,2))
        pre_path[:,0] = np.array(x_pred_vals)
        pre_path[:,1] = np.array(y_pred_vals)        
        v = self.iN.uv[0]()
        w = self.iN.uw[0]()                                     
        return pre_path, v, w      
        
        
class Turtlebot_core():
    def __init__(self):
        rospy.init_node("Turtlebot_core", anonymous=True)
        self.listener = tf.TransformListener()
        rospy.Subscriber("/odom", Odometry, self.odomCallback)
        self.pub_refpath = rospy.Publisher("/ref_path", Path, queue_size=1)
        self.pub_prepath = rospy.Publisher("/pre_path", Path, queue_size=1)
        self.pub_cmd = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=1)
        self.rp = np.zeros(3) 
        self.crv = 0.0
        self.crw = 0.0
        self.mpc = MPC() 
        rate = rospy.Rate(10) # 10HZ
        while not rospy.is_shutdown():
            self.getrobotpose()
            cwps = getcwps(self.rp[0:2])
            px = self.rp[0] + self.crv*np.cos(self.rp[2])*0.1
            py = self.rp[1] + self.crw*np.sin(self.rp[2])*0.1
            psi = self.rp[2] + self.crw*0.1
            
            self.rp[0] = px
            self.rp[1] = py
            self.rp[2] = psi
            
            cwps_robot = np.zeros((len(cwps), 2))
            
            for i in range(len(cwps)):
                dx = cwps[i,0] - px
                dy = cwps[i,1] - py
                
                cwps_robot[i,0] = dx*np.cos(psi) + dy*np.sin(psi)
                cwps_robot[i,1] = dy*np.cos(psi) - dx*np.sin(psi)
                
            coeffs = np.polyfit(cwps_robot[:,0], cwps_robot[:,1], 3)
            print(coeffs)
            cte = cubic_fun(coeffs, 0)
            
            f_prime_x = coeffs[2]
            epsi = np.arctan(f_prime_x)
            s0 = np.array([0.0, 0.0, 0.0, cte, epsi])
            pre_path, v, w = self.mpc.Solve(s0, coeffs)
            self.pub_ref_path(cwps_robot)
            self.pub_pre_path(pre_path)
            self.pub_Command(v, w)
            print(pre_path)
            print(v,w)
            plot_durations(cwps, pre_path[:,0], pre_path[:,1])
            rate.sleep()        
        rospy.spin()            
            
    def getrobotpose(self):
        try:
            (trans,rot) = self.listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return   
        self.rp[0] = trans[0]
        self.rp[1] = trans[1]
        r,p,y = tf.transformations.euler_from_quaternion(rot)
        self.rp[2] = y
        
    def odomCallback(self, data):
        self.crv = data.twist.twist.linear.x
        self.crw = data.twist.twist.angular.z
        
    def pub_ref_path(self, ref_path):        
        msg_ref_path = Path()
        msg_ref_path.header.stamp = rospy.Time.now()
        msg_ref_path.header.frame_id = "base_link"
        for i in range(len(ref_path)):
            pose = PoseStamped()
            pose.pose.position.x = ref_path[i,0]
            pose.pose.position.y = ref_path[i,1]
            msg_ref_path.poses.append(copy.deepcopy(pose))
            
        self.pub_refpath.publish(msg_ref_path)
            
    def pub_pre_path(self, pre_path):
        msg_pre_path = Path()
        msg_pre_path.header.stamp = rospy.Time.now()
        msg_pre_path.header.frame_id = "base_link"
        for i in range(len(pre_path)):
            pose = PoseStamped()
            pose.pose.position.x = pre_path[i,0]
            pose.pose.position.y = pre_path[i,1]
            msg_pre_path.poses.append(copy.deepcopy(pose))    
        self.pub_prepath.publish(msg_pre_path)
        
    def pub_Command(self, v, w):
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.pub_cmd.publish(twist)     
                
if __name__ == "__main__":
    turtlebot_core = Turtlebot_core() 


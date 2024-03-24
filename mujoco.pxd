
cdef extern from "mujoco/mjmodel.h" nogil:
    ctypedef double mjtNum
    ctypedef unsigned char mjtByte

    # global constants
    enum: mjPI
    enum: mjMAXVAL             # maximum value in qpos, qvel, qacc
    enum: mjMINMU               # minimum friction coefficient
    enum: mjMINIMP            # minimum constraint impedance
    enum: mjMAXIMP            # maximum constraint impedance
    enum: mjMAXCONPAIR            # maximum number of contacts per geom pair
    enum: mjMAXTREEDEPTH          # maximum bounding volume hierarchy depth
    enum: mjMAXVFS              # maximum number of files in virtual file system
    enum: mjMAXVFSNAME          # maximum filename size in virtual file system


    #---------------------------------- sizes ---------------------------------------------------------

    enum: mjNEQDATA               # number of eq_data fields
    enum: mjNDYN                  # number of actuator dynamics parameters
    enum: mjNGAIN                 # number of actuator gain parameters
    enum: mjNBIAS                 # number of actuator bias parameters
    enum: mjNFLUID                # number of fluid interaction parameters
    enum: mjNREF                   # number of solver reference parameters
    enum: mjNIMP                   # number of solver impedance parameters
    enum: mjNSOLVER              # size of one mjData.solver array
    enum: mjNISLAND               # number of mjData.solver arrays


    ctypedef struct mjOption:                # physics options
        # timing parameters
        mjtNum timestep                # timestep
        mjtNum apirate                 # update rate for remote API (Hz)

        # solver parameters
        mjtNum impratio                # ratio of friction-to-normal contact impedance
        mjtNum tolerance               # main solver tolerance
        mjtNum ls_tolerance            # CG/Newton linesearch tolerance
        mjtNum noslip_tolerance        # noslip solver tolerance
        mjtNum mpr_tolerance           # MPR solver tolerance

        # physical constants
        mjtNum gravity[3]              # gravitational acceleration
        mjtNum wind[3]                 # wind (for lift, drag and viscosity)
        mjtNum magnetic[3]             # global magnetic flux
        mjtNum density                 # density of medium
        mjtNum viscosity               # viscosity of medium

        # override contact solver parameters (if enabled)
        mjtNum o_margin                # margin
        mjtNum o_solref[mjNREF]        # solref
        mjtNum o_solimp[mjNIMP]        # solimp
        mjtNum o_friction[5]           # friction

        # discrete settings
        int integrator                 # integration mode (mjtIntegrator)
        int cone                       # type of friction cone (mjtCone)
        int jacobian                   # type of Jacobian (mjtJacobian)
        int solver                     # solver algorithm (mjtSolver)
        int iterations                 # maximum number of main solver iterations
        int ls_iterations              # maximum number of CG/Newton linesearch iterations
        int noslip_iterations          # maximum number of noslip solver iterations
        int mpr_iterations             # maximum number of MPR solver iterations
        int disableflags               # bit flags for disabling standard features
        int enableflags                # bit flags for enabling optional features
        int disableactuator            # bit flags for disabling actuators by group id

        # sdf collision settings
        int sdf_initpoints             # number of starting points for gradient descent
        int sdf_iterations             # max number of iterations for gradient descent


    ctypedef struct mjVisual:                # visualization options
        pass


    ctypedef struct mjStatistic:             # model statistics (in qpos0)
        mjtNum meaninertia             # mean diagonal inertia
        mjtNum meanmass                # mean body mass
        mjtNum meansize                # mean body size
        mjtNum extent                  # spatial extent
        mjtNum center[3]               # center of model


    ctypedef struct mjModel:
        # ------------------------------- sizes

        # sizes needed at mjModel construction
        int nq                         # number of generalized coordinates = dim(qpos)
        int nv                         # number of degrees of freedom = dim(qvel)
        int nu                         # number of actuators/controls = dim(ctrl)
        int na                         # number of activation states = dim(act)
        int nbody                      # number of bodies
        int nbvh                       # number of total bounding volumes in all bodies
        int nbvhstatic                 # number of static bounding volumes (aabb stored in mjModel)
        int nbvhdynamic                # number of dynamic bounding volumes (aabb stored in mjData)
        int njnt                       # number of joints
        int ngeom                      # number of geoms
        int nsite                      # number of sites
        int ncam                       # number of cameras
        int nlight                     # number of lights
        int nflex                      # number of flexes
        int nflexvert                  # number of vertices in all flexes
        int nflexedge                  # number of edges in all flexes
        int nflexelem                  # number of elements in all flexes
        int nflexelemdata              # number of element vertex ids in all flexes
        int nflexshelldata             # number of shell fragment vertex ids in all flexes
        int nflexevpair                # number of element-vertex pairs in all flexes
        int nflextexcoord              # number of vertices with texture coordinates
        int nmesh                      # number of meshes
        int nmeshvert                  # number of vertices in all meshes
        int nmeshnormal                # number of normals in all meshes
        int nmeshtexcoord              # number of texcoords in all meshes
        int nmeshface                  # number of triangular faces in all meshes
        int nmeshgraph                 # number of ints in mesh auxiliary data
        int nskin                      # number of skins
        int nskinvert                  # number of vertices in all skins
        int nskintexvert               # number of vertiex with texcoords in all skins
        int nskinface                  # number of triangular faces in all skins
        int nskinbone                  # number of bones in all skins
        int nskinbonevert              # number of vertices in all skin bones
        int nhfield                    # number of heightfields
        int nhfielddata                # number of data points in all heightfields
        int ntex                       # number of textures
        int ntexdata                   # number of bytes in texture rgb data
        int nmat                       # number of materials
        int npair                      # number of predefined geom pairs
        int nexclude                   # number of excluded geom pairs
        int neq                        # number of equality constraints
        int ntendon                    # number of tendons
        int nwrap                      # number of wrap objects in all tendon paths
        int nsensor                    # number of sensors
        int nnumeric                   # number of numeric custom fields
        int nnumericdata               # number of mjtNums in all numeric fields
        int ntext                      # number of text custom fields
        int ntextdata                  # number of mjtBytes in all text fields
        int ntuple                     # number of tuple custom fields
        int ntupledata                 # number of objects in all tuple fields
        int nkey                       # number of keyframes
        int nmocap                     # number of mocap bodies
        int nplugin                    # number of plugin instances
        int npluginattr                # number of chars in all plugin config attributes
        int nuser_body                 # number of mjtNums in body_user
        int nuser_jnt                  # number of mjtNums in jnt_user
        int nuser_geom                 # number of mjtNums in geom_user
        int nuser_site                 # number of mjtNums in site_user
        int nuser_cam                  # number of mjtNums in cam_user
        int nuser_tendon               # number of mjtNums in tendon_user
        int nuser_actuator             # number of mjtNums in actuator_user
        int nuser_sensor               # number of mjtNums in sensor_user
        int nnames                     # number of chars in all names
        int nnames_map                 # number of slots in the names hash map
        int npaths                     # number of chars in all paths

        # sizes set after mjModel construction (only affect mjData)
        int nM                         # number of non-zeros in sparse inertia matrix
        int nD                         # number of non-zeros in sparse dof-dof matrix
        int nB                         # number of non-zeros in sparse body-dof matrix
        int ntree                      # number of kinematic trees under world body
        int nemax                      # number of potential equality-constraint rows
        int njmax                      # number of available rows in constraint Jacobian
        int nconmax                    # number of potential contacts in contact list
        int nuserdata                  # number of extra fields in mjData
        int nsensordata                # number of fields in sensor data vector
        int npluginstate               # number of fields in plugin state vector

        size_t narena                  # number of bytes in the mjData arena (inclusive of stack)
        size_t nbuffer                 # number of bytes in buffer

        # ------------------------------- options and statistics

        mjOption opt                   # physics options
        mjVisual vis                   # visualization options
        mjStatistic stat               # model statistics

        # ------------------------------- buffers

        # main buffer
        void*     buffer               # main buffer all pointers point in it    (nbuffer)

        # default generalized coordinates
        mjtNum*   qpos0                # qpos values at default pose              (nq x 1)
        mjtNum*   qpos_spring          # reference pose for springs               (nq x 1)

        # bodies
        int*      body_parentid        # id of body's parent                      (nbody x 1)
        int*      body_rootid          # id of root above body                    (nbody x 1)
        int*      body_weldid          # id of body that this body is welded to   (nbody x 1)
        int*      body_mocapid         # id of mocap data -1: none               (nbody x 1)
        int*      body_jntnum          # number of joints for this body           (nbody x 1)
        int*      body_jntadr          # start addr of joints -1: no joints      (nbody x 1)
        int*      body_dofnum          # number of motion degrees of freedom      (nbody x 1)
        int*      body_dofadr          # start addr of dofs -1: no dofs          (nbody x 1)
        int*      body_treeid          # id of body's kinematic tree -1: static  (nbody x 1)
        int*      body_geomnum         # number of geoms                          (nbody x 1)
        int*      body_geomadr         # start addr of geoms -1: no geoms        (nbody x 1)
        mjtByte*  body_simple          # 1: diag M 2: diag M, sliders only       (nbody x 1)
        mjtByte*  body_sameframe       # inertial frame is same as body frame     (nbody x 1)
        mjtNum*   body_pos             # position offset rel. to parent body      (nbody x 3)
        mjtNum*   body_quat            # orientation offset rel. to parent body   (nbody x 4)
        mjtNum*   body_ipos            # local position of center of mass         (nbody x 3)
        mjtNum*   body_iquat           # local orientation of inertia ellipsoid   (nbody x 4)
        mjtNum*   body_mass            # mass                                     (nbody x 1)
        mjtNum*   body_subtreemass     # mass of subtree starting at this body    (nbody x 1)
        mjtNum*   body_inertia         # diagonal inertia in ipos/iquat frame     (nbody x 3)
        mjtNum*   body_invweight0      # mean inv inert in qpos0 (trn, rot)       (nbody x 2)
        mjtNum*   body_gravcomp        # antigravity force, units of body weight  (nbody x 1)
        mjtNum*   body_margin          # MAX over all geom margins                (nbody x 1)
        mjtNum*   body_user            # user data                                (nbody x nuser_body)
        int*      body_plugin          # plugin instance id -1: not in use       (nbody x 1)
        int*      body_contype         # OR over all geom contypes                (nbody x 1)
        int*      body_conaffinity     # OR over all geom conaffinities           (nbody x 1)
        int*      body_bvhadr          # address of bvh root                      (nbody x 1)
        int*      body_bvhnum          # number of bounding volumes               (nbody x 1)

        # bounding volume hierarchy
        int*      bvh_depth            # depth in the bounding volume hierarchy   (nbvh x 1)
        int*      bvh_child            # left and right children in tree          (nbvh x 2)
        int*      bvh_nodeid           # geom or elem id of node -1: non-leaf    (nbvh x 1)
        mjtNum*   bvh_aabb             # local bounding box (center, size)        (nbvhstatic x 6)

        # joints
        int*      jnt_type             # type of joint (mjtJoint)                 (njnt x 1)
        int*      jnt_qposadr          # start addr in 'qpos' for joint's data    (njnt x 1)
        int*      jnt_dofadr           # start addr in 'qvel' for joint's data    (njnt x 1)
        int*      jnt_bodyid           # id of joint's body                       (njnt x 1)
        int*      jnt_group            # group for visibility                     (njnt x 1)
        mjtByte*  jnt_limited          # does joint have limits                   (njnt x 1)
        mjtByte*  jnt_actfrclimited    # does joint have actuator force limits    (njnt x 1)
        mjtNum*   jnt_solref           # constraint solver reference: limit       (njnt x mjNREF)
        mjtNum*   jnt_solimp           # constraint solver impedance: limit       (njnt x mjNIMP)
        mjtNum*   jnt_pos              # local anchor position                    (njnt x 3)
        mjtNum*   jnt_axis             # local joint axis                         (njnt x 3)
        mjtNum*   jnt_stiffness        # stiffness coefficient                    (njnt x 1)
        mjtNum*   jnt_range            # joint limits                             (njnt x 2)
        mjtNum*   jnt_actfrcrange      # range of total actuator force            (njnt x 2)
        mjtNum*   jnt_margin           # min distance for limit detection         (njnt x 1)
        mjtNum*   jnt_user             # user data                                (njnt x nuser_jnt)

        # dofs
        int*      dof_bodyid           # id of dof's body                         (nv x 1)
        int*      dof_jntid            # id of dof's joint                        (nv x 1)
        int*      dof_parentid         # id of dof's parent -1: none             (nv x 1)
        int*      dof_treeid           # id of dof's kinematic tree               (nv x 1)
        int*      dof_Madr             # dof address in M-diagonal                (nv x 1)
        int*      dof_simplenum        # number of consecutive simple dofs        (nv x 1)
        mjtNum*   dof_solref           # constraint solver reference:frictionloss (nv x mjNREF)
        mjtNum*   dof_solimp           # constraint solver impedance:frictionloss (nv x mjNIMP)
        mjtNum*   dof_frictionloss     # dof friction loss                        (nv x 1)
        mjtNum*   dof_armature         # dof armature inertia/mass                (nv x 1)
        mjtNum*   dof_damping          # damping coefficient                      (nv x 1)
        mjtNum*   dof_invweight0       # diag. inverse inertia in qpos0           (nv x 1)
        mjtNum*   dof_M0               # diag. inertia in qpos0                   (nv x 1)

        # geoms
        int*      geom_type            # geometric type (mjtGeom)                 (ngeom x 1)
        int*      geom_contype         # geom contact type                        (ngeom x 1)
        int*      geom_conaffinity     # geom contact affinity                    (ngeom x 1)
        int*      geom_condim          # contact dimensionality (1, 3, 4, 6)      (ngeom x 1)
        int*      geom_bodyid          # id of geom's body                        (ngeom x 1)
        int*      geom_dataid          # id of geom's mesh/hfield -1: none       (ngeom x 1)
        int*      geom_matid           # material id for rendering -1: none      (ngeom x 1)
        int*      geom_group           # group for visibility                     (ngeom x 1)
        int*      geom_priority        # geom contact priority                    (ngeom x 1)
        int*      geom_plugin          # plugin instance id -1: not in use       (ngeom x 1)
        mjtByte*  geom_sameframe       # same as body frame (1) or iframe (2)     (ngeom x 1)
        mjtNum*   geom_solmix          # mixing coef for solref/imp in geom pair  (ngeom x 1)
        mjtNum*   geom_solref          # constraint solver reference: contact     (ngeom x mjNREF)
        mjtNum*   geom_solimp          # constraint solver impedance: contact     (ngeom x mjNIMP)
        mjtNum*   geom_size            # geom-specific size parameters            (ngeom x 3)
        mjtNum*   geom_aabb            # bounding box, (center, size)             (ngeom x 6)
        mjtNum*   geom_rbound          # radius of bounding sphere                (ngeom x 1)
        mjtNum*   geom_pos             # local position offset rel. to body       (ngeom x 3)
        mjtNum*   geom_quat            # local orientation offset rel. to body    (ngeom x 4)
        mjtNum*   geom_friction        # friction for (slide, spin, roll)         (ngeom x 3)
        mjtNum*   geom_margin          # detect contact if dist<margin            (ngeom x 1)
        mjtNum*   geom_gap             # include in solver if dist<margin-gap     (ngeom x 1)
        mjtNum*   geom_fluid           # fluid interaction parameters             (ngeom x mjNFLUID)
        mjtNum*   geom_user            # user data                                (ngeom x nuser_geom)
        float*    geom_rgba            # rgba when material is omitted            (ngeom x 4)

        # sites
        int*      site_type            # geom type for rendering (mjtGeom)        (nsite x 1)
        int*      site_bodyid          # id of site's body                        (nsite x 1)
        int*      site_matid           # material id for rendering -1: none      (nsite x 1)
        int*      site_group           # group for visibility                     (nsite x 1)
        mjtByte*  site_sameframe       # same as body frame (1) or iframe (2)     (nsite x 1)
        mjtNum*   site_size            # geom size for rendering                  (nsite x 3)
        mjtNum*   site_pos             # local position offset rel. to body       (nsite x 3)
        mjtNum*   site_quat            # local orientation offset rel. to body    (nsite x 4)
        mjtNum*   site_user            # user data                                (nsite x nuser_site)
        float*    site_rgba            # rgba when material is omitted            (nsite x 4)

        # cameras
        int*      cam_mode             # camera tracking mode (mjtCamLight)       (ncam x 1)
        int*      cam_bodyid           # id of camera's body                      (ncam x 1)
        int*      cam_targetbodyid     # id of targeted body -1: none            (ncam x 1)
        mjtNum*   cam_pos              # position rel. to body frame              (ncam x 3)
        mjtNum*   cam_quat             # orientation rel. to body frame           (ncam x 4)
        mjtNum*   cam_poscom0          # global position rel. to sub-com in qpos0 (ncam x 3)
        mjtNum*   cam_pos0             # global position rel. to body in qpos0    (ncam x 3)
        mjtNum*   cam_mat0             # global orientation in qpos0              (ncam x 9)
        int*      cam_resolution       # [width, height] in pixels                (ncam x 2)
        mjtNum*   cam_fovy             # y-field of view (deg)                    (ncam x 1)
        float*    cam_intrinsic        # [focal length principal point]          (ncam x 4)
        float*    cam_sensorsize       # sensor size                              (ncam x 2)
        mjtNum*   cam_ipd              # inter-pupilary distance                  (ncam x 1)
        mjtNum*   cam_user             # user data                                (ncam x nuser_cam)

        # lights
        int*      light_mode           # light tracking mode (mjtCamLight)        (nlight x 1)
        int*      light_bodyid         # id of light's body                       (nlight x 1)
        int*      light_targetbodyid   # id of targeted body -1: none            (nlight x 1)
        mjtByte*  light_directional    # directional light                        (nlight x 1)
        mjtByte*  light_castshadow     # does light cast shadows                  (nlight x 1)
        mjtByte*  light_active         # is light on                              (nlight x 1)
        mjtNum*   light_pos            # position rel. to body frame              (nlight x 3)
        mjtNum*   light_dir            # direction rel. to body frame             (nlight x 3)
        mjtNum*   light_poscom0        # global position rel. to sub-com in qpos0 (nlight x 3)
        mjtNum*   light_pos0           # global position rel. to body in qpos0    (nlight x 3)
        mjtNum*   light_dir0           # global direction in qpos0                (nlight x 3)
        float*    light_attenuation    # OpenGL attenuation (quadratic model)     (nlight x 3)
        float*    light_cutoff         # OpenGL cutoff                            (nlight x 1)
        float*    light_exponent       # OpenGL exponent                          (nlight x 1)
        float*    light_ambient        # ambient rgb (alpha=1)                    (nlight x 3)
        float*    light_diffuse        # diffuse rgb (alpha=1)                    (nlight x 3)
        float*    light_specular       # specular rgb (alpha=1)                   (nlight x 3)

        # flexes: contact properties
        int*      flex_contype         # flex contact type                        (nflex x 1)
        int*      flex_conaffinity     # flex contact affinity                    (nflex x 1)
        int*      flex_condim          # contact dimensionality (1, 3, 4, 6)      (nflex x 1)
        int*      flex_priority        # flex contact priority                    (nflex x 1)
        mjtNum*   flex_solmix          # mix coef for solref/imp in contact pair  (nflex x 1)
        mjtNum*   flex_solref          # constraint solver reference: contact     (nflex x mjNREF)
        mjtNum*   flex_solimp          # constraint solver impedance: contact     (nflex x mjNIMP)
        mjtNum*   flex_friction        # friction for (slide, spin, roll)         (nflex x 3)
        mjtNum*   flex_margin          # detect contact if dist<margin            (nflex x 1)
        mjtNum*   flex_gap             # include in solver if dist<margin-gap     (nflex x 1)
        mjtByte*  flex_internal        # internal flex collision enabled          (nflex x 1)
        int*      flex_selfcollide     # self collision mode (mjtFlexSelf)        (nflex x 1)
        int*      flex_activelayers    # number of active element layers, 3D only (nflex x 1)

        # flexes: other properties
        int*      flex_dim             # 1: lines, 2: triangles, 3: tetrahedra    (nflex x 1)
        int*      flex_matid           # material id for rendering                (nflex x 1)
        int*      flex_group           # group for visibility                     (nflex x 1)
        int*      flex_vertadr         # first vertex address                     (nflex x 1)
        int*      flex_vertnum         # number of vertices                       (nflex x 1)
        int*      flex_edgeadr         # first edge address                       (nflex x 1)
        int*      flex_edgenum         # number of edges                          (nflex x 1)
        int*      flex_elemadr         # first element address                    (nflex x 1)
        int*      flex_elemnum         # number of elements                       (nflex x 1)
        int*      flex_elemdataadr     # first element vertex id address          (nflex x 1)
        int*      flex_shellnum        # number of shells                         (nflex x 1)
        int*      flex_shelldataadr    # first shell data address                 (nflex x 1)
        int*      flex_evpairadr       # first evpair address                     (nflex x 1)
        int*      flex_evpairnum       # number of evpairs                        (nflex x 1)
        int*      flex_texcoordadr     # address in flex_texcoord -1: none       (nflex x 1)
        int*      flex_vertbodyid      # vertex body ids                          (nflexvert x 1)
        int*      flex_edge            # edge vertex ids (2 per edge)             (nflexedge x 2)
        int*      flex_elem            # element vertex ids (dim+1 per elem)      (nflexelemdata x 1)
        int*      flex_elemlayer       # element distance from surface, 3D only   (nflexelem x 1)
        int*      flex_shell           # shell fragment vertex ids (dim per frag) (nflexshelldata x 1)
        int*      flex_evpair          # (element, vertex) collision pairs        (nflexevpair x 2)
        mjtNum*   flex_vert            # vertex positions in local body frames    (nflexvert x 3)
        mjtNum*   flex_xvert0          # Cartesian vertex positions in qpos0      (nflexvert x 3)
        mjtNum*   flexedge_length0     # edge lengths in qpos0                    (nflexedge x 1)
        mjtNum*   flexedge_invweight0  # edge inv. weight in qpos0                (nflexedge x 1)
        mjtNum*   flex_radius          # radius around primitive element          (nflex x 1)
        mjtNum*   flex_edgestiffness   # edge stiffness                           (nflex x 1)
        mjtNum*   flex_edgedamping     # edge damping                             (nflex x 1)
        mjtByte*  flex_edgeequality    # is edge equality constraint defined      (nflex x 1)
        mjtByte*  flex_rigid           # are all verices in the same body         (nflex x 1)
        mjtByte*  flexedge_rigid       # are both edge vertices in same body      (nflexedge x 1)
        mjtByte*  flex_centered        # are all vertex coordinates (0,0,0)       (nflex x 1)
        mjtByte*  flex_flatskin        # render flex skin with flat shading       (nflex x 1)
        int*      flex_bvhadr          # address of bvh root -1: no bvh          (nflex x 1)
        int*      flex_bvhnum          # number of bounding volumes               (nflex x 1)
        float*    flex_rgba            # rgba when material is omitted            (nflex x 4)
        float*    flex_texcoord        # vertex texture coordinates               (nflextexcoord x 2)

        # meshes
        int*      mesh_vertadr         # first vertex address                     (nmesh x 1)
        int*      mesh_vertnum         # number of vertices                       (nmesh x 1)
        int*      mesh_faceadr         # first face address                       (nmesh x 1)
        int*      mesh_facenum         # number of faces                          (nmesh x 1)
        int*      mesh_bvhadr          # address of bvh root                      (nmesh x 1)
        int*      mesh_bvhnum          # number of bvh                            (nmesh x 1)
        int*      mesh_normaladr       # first normal address                     (nmesh x 1)
        int*      mesh_normalnum       # number of normals                        (nmesh x 1)
        int*      mesh_texcoordadr     # texcoord data address -1: no texcoord   (nmesh x 1)
        int*      mesh_texcoordnum     # number of texcoord                       (nmesh x 1)
        int*      mesh_graphadr        # graph data address -1: no graph         (nmesh x 1)
        float*    mesh_vert            # vertex positions for all meshes          (nmeshvert x 3)
        float*    mesh_normal          # normals for all meshes                   (nmeshnormal x 3)
        float*    mesh_texcoord        # vertex texcoords for all meshes          (nmeshtexcoord x 2)
        int*      mesh_face            # vertex face data                         (nmeshface x 3)
        int*      mesh_facenormal      # normal face data                         (nmeshface x 3)
        int*      mesh_facetexcoord    # texture face data                        (nmeshface x 3)
        int*      mesh_graph           # convex graph data                        (nmeshgraph x 1)
        mjtNum*   mesh_pos             # translation applied to asset vertices    (nmesh x 3)
        mjtNum*   mesh_quat            # rotation applied to asset vertices       (nmesh x 4)
        int*      mesh_pathadr         # address of asset path for mesh -1: none (nmesh x 1)

        # skins
        int*      skin_matid           # skin material id -1: none               (nskin x 1)
        int*      skin_group           # group for visibility                     (nskin x 1)
        float*    skin_rgba            # skin rgba                                (nskin x 4)
        float*    skin_inflate         # inflate skin in normal direction         (nskin x 1)
        int*      skin_vertadr         # first vertex address                     (nskin x 1)
        int*      skin_vertnum         # number of vertices                       (nskin x 1)
        int*      skin_texcoordadr     # texcoord data address -1: no texcoord   (nskin x 1)
        int*      skin_faceadr         # first face address                       (nskin x 1)
        int*      skin_facenum         # number of faces                          (nskin x 1)
        int*      skin_boneadr         # first bone in skin                       (nskin x 1)
        int*      skin_bonenum         # number of bones in skin                  (nskin x 1)
        float*    skin_vert            # vertex positions for all skin meshes     (nskinvert x 3)
        float*    skin_texcoord        # vertex texcoords for all skin meshes     (nskintexvert x 2)
        int*      skin_face            # triangle faces for all skin meshes       (nskinface x 3)
        int*      skin_bonevertadr     # first vertex in each bone                (nskinbone x 1)
        int*      skin_bonevertnum     # number of vertices in each bone          (nskinbone x 1)
        float*    skin_bonebindpos     # bind pos of each bone                    (nskinbone x 3)
        float*    skin_bonebindquat    # bind quat of each bone                   (nskinbone x 4)
        int*      skin_bonebodyid      # body id of each bone                     (nskinbone x 1)
        int*      skin_bonevertid      # mesh ids of vertices in each bone        (nskinbonevert x 1)
        float*    skin_bonevertweight  # weights of vertices in each bone         (nskinbonevert x 1)
        int*      skin_pathadr         # address of asset path for skin -1: none (nskin x 1)

        # height fields
        mjtNum*   hfield_size          # (x, y, z_top, z_bottom)                    (nhfield x 4)
        int*      hfield_nrow          # number of rows in grid                     (nhfield x 1)
        int*      hfield_ncol          # number of columns in grid                  (nhfield x 1)
        int*      hfield_adr           # address in hfield_data                     (nhfield x 1)
        float*    hfield_data          # elevation data                             (nhfielddata x 1)
        int*      hfield_pathadr       # address of asset path for hfield -1: none (nhfield x 1)

        # textures
        int*      tex_type             # texture type (mjtTexture)                  (ntex x 1)
        int*      tex_height           # number of rows in texture image            (ntex x 1)
        int*      tex_width            # number of columns in texture image         (ntex x 1)
        int*      tex_adr              # address in rgb                             (ntex x 1)
        mjtByte*  tex_rgb              # rgb (alpha = 1)                            (ntexdata x 1)
        int*      tex_pathadr         # address of asset path for texture -1: none (ntex x 1)

        # materials
        int*      mat_texid            # texture id -1: none                     (nmat x 1)
        mjtByte*  mat_texuniform       # make texture cube uniform                (nmat x 1)
        float*    mat_texrepeat        # texture repetition for 2d mapping        (nmat x 2)
        float*    mat_emission         # emission (x rgb)                         (nmat x 1)
        float*    mat_specular         # specular (x white)                       (nmat x 1)
        float*    mat_shininess        # shininess coef                           (nmat x 1)
        float*    mat_reflectance      # reflectance (0: disable)                 (nmat x 1)
        float*    mat_rgba             # rgba                                     (nmat x 4)

        # predefined geom pairs for collision detection has precedence over exclude
        int*      pair_dim             # contact dimensionality                   (npair x 1)
        int*      pair_geom1           # id of geom1                              (npair x 1)
        int*      pair_geom2           # id of geom2                              (npair x 1)
        int*      pair_signature       # body1 << 16 + body2                      (npair x 1)
        mjtNum*   pair_solref          # solver reference: contact normal         (npair x mjNREF)
        mjtNum*   pair_solreffriction  # solver reference: contact friction       (npair x mjNREF)
        mjtNum*   pair_solimp          # solver impedance: contact                (npair x mjNIMP)
        mjtNum*   pair_margin          # detect contact if dist<margin            (npair x 1)
        mjtNum*   pair_gap             # include in solver if dist<margin-gap     (npair x 1)
        mjtNum*   pair_friction        # tangent1, 2, spin, roll1, 2              (npair x 5)

        # excluded body pairs for collision detection
        int*      exclude_signature    # body1 << 16 + body2                      (nexclude x 1)

        # equality constraints
        int*      eq_type              # constraint type (mjtEq)                  (neq x 1)
        int*      eq_obj1id            # id of object 1                           (neq x 1)
        int*      eq_obj2id            # id of object 2                           (neq x 1)
        mjtByte*  eq_active0           # initial enable/disable constraint state  (neq x 1)
        mjtNum*   eq_solref            # constraint solver reference              (neq x mjNREF)
        mjtNum*   eq_solimp            # constraint solver impedance              (neq x mjNIMP)
        mjtNum*   eq_data              # numeric data for constraint              (neq x mjNEQDATA)

        # tendons
        int*      tendon_adr           # address of first object in tendon's path (ntendon x 1)
        int*      tendon_num           # number of objects in tendon's path       (ntendon x 1)
        int*      tendon_matid         # material id for rendering                (ntendon x 1)
        int*      tendon_group         # group for visibility                     (ntendon x 1)
        mjtByte*  tendon_limited       # does tendon have length limits           (ntendon x 1)
        mjtNum*   tendon_width         # width for rendering                      (ntendon x 1)
        mjtNum*   tendon_solref_lim    # constraint solver reference: limit       (ntendon x mjNREF)
        mjtNum*   tendon_solimp_lim    # constraint solver impedance: limit       (ntendon x mjNIMP)
        mjtNum*   tendon_solref_fri    # constraint solver reference: friction    (ntendon x mjNREF)
        mjtNum*   tendon_solimp_fri    # constraint solver impedance: friction    (ntendon x mjNIMP)
        mjtNum*   tendon_range         # tendon length limits                     (ntendon x 2)
        mjtNum*   tendon_margin        # min distance for limit detection         (ntendon x 1)
        mjtNum*   tendon_stiffness     # stiffness coefficient                    (ntendon x 1)
        mjtNum*   tendon_damping       # damping coefficient                      (ntendon x 1)
        mjtNum*   tendon_frictionloss  # loss due to friction                     (ntendon x 1)
        mjtNum*   tendon_lengthspring  # spring resting length range              (ntendon x 2)
        mjtNum*   tendon_length0       # tendon length in qpos0                   (ntendon x 1)
        mjtNum*   tendon_invweight0    # inv. weight in qpos0                     (ntendon x 1)
        mjtNum*   tendon_user          # user data                                (ntendon x nuser_tendon)
        float*    tendon_rgba          # rgba when material is omitted            (ntendon x 4)

        # list of all wrap objects in tendon paths
        int*      wrap_type            # wrap object type (mjtWrap)               (nwrap x 1)
        int*      wrap_objid           # object id: geom, site, joint             (nwrap x 1)
        mjtNum*   wrap_prm             # divisor, joint coef, or site id          (nwrap x 1)

        # actuators
        int*      actuator_trntype     # transmission type (mjtTrn)               (nu x 1)
        int*      actuator_dyntype     # dynamics type (mjtDyn)                   (nu x 1)
        int*      actuator_gaintype    # gain type (mjtGain)                      (nu x 1)
        int*      actuator_biastype    # bias type (mjtBias)                      (nu x 1)
        int*      actuator_trnid       # transmission id: joint, tendon, site     (nu x 2)
        int*      actuator_actadr      # first activation address -1: stateless  (nu x 1)
        int*      actuator_actnum      # number of activation variables           (nu x 1)
        int*      actuator_group       # group for visibility                     (nu x 1)
        mjtByte*  actuator_ctrllimited # is control limited                       (nu x 1)
        mjtByte*  actuator_forcelimited# is force limited                         (nu x 1)
        mjtByte*  actuator_actlimited  # is activation limited                    (nu x 1)
        mjtNum*   actuator_dynprm      # dynamics parameters                      (nu x mjNDYN)
        mjtNum*   actuator_gainprm     # gain parameters                          (nu x mjNGAIN)
        mjtNum*   actuator_biasprm     # bias parameters                          (nu x mjNBIAS)
        mjtByte*  actuator_actearly    # step activation before force             (nu x 1)
        mjtNum*   actuator_ctrlrange   # range of controls                        (nu x 2)
        mjtNum*   actuator_forcerange  # range of forces                          (nu x 2)
        mjtNum*   actuator_actrange    # range of activations                     (nu x 2)
        mjtNum*   actuator_gear        # scale length and transmitted force       (nu x 6)
        mjtNum*   actuator_cranklength # crank length for slider-crank            (nu x 1)
        mjtNum*   actuator_acc0        # acceleration from unit force in qpos0    (nu x 1)
        mjtNum*   actuator_length0     # actuator length in qpos0                 (nu x 1)
        mjtNum*   actuator_lengthrange # feasible actuator length range           (nu x 2)
        mjtNum*   actuator_user        # user data                                (nu x nuser_actuator)
        int*      actuator_plugin      # plugin instance id -1: not a plugin     (nu x 1)

        # sensors
        int*      sensor_type          # sensor type (mjtSensor)                  (nsensor x 1)
        int*      sensor_datatype      # numeric data type (mjtDataType)          (nsensor x 1)
        int*      sensor_needstage     # required compute stage (mjtStage)        (nsensor x 1)
        int*      sensor_objtype       # type of sensorized object (mjtObj)       (nsensor x 1)
        int*      sensor_objid         # id of sensorized object                  (nsensor x 1)
        int*      sensor_reftype       # type of reference frame (mjtObj)         (nsensor x 1)
        int*      sensor_refid         # id of reference frame -1: global frame  (nsensor x 1)
        int*      sensor_dim           # number of scalar outputs                 (nsensor x 1)
        int*      sensor_adr           # address in sensor array                  (nsensor x 1)
        mjtNum*   sensor_cutoff        # cutoff for real and positive 0: ignore  (nsensor x 1)
        mjtNum*   sensor_noise         # noise standard deviation                 (nsensor x 1)
        mjtNum*   sensor_user          # user data                                (nsensor x nuser_sensor)
        int*      sensor_plugin        # plugin instance id -1: not a plugin     (nsensor x 1)

        # plugin instances
        int*      plugin               # globally registered plugin slot number   (nplugin x 1)
        int*      plugin_stateadr      # address in the plugin state array        (nplugin x 1)
        int*      plugin_statenum      # number of states in the plugin instance  (nplugin x 1)
        char*     plugin_attr          # config attributes of plugin instances    (npluginattr x 1)
        int*      plugin_attradr       # address to each instance's config attrib (nplugin x 1)

        # custom numeric fields
        int*      numeric_adr          # address of field in numeric_data         (nnumeric x 1)
        int*      numeric_size         # size of numeric field                    (nnumeric x 1)
        mjtNum*   numeric_data         # array of all numeric fields              (nnumericdata x 1)

        # custom text fields
        int*      text_adr             # address of text in text_data             (ntext x 1)
        int*      text_size            # size of text field (strlen+1)            (ntext x 1)
        char*     text_data            # array of all text fields (0-terminated)  (ntextdata x 1)

        # custom tuple fields
        int*      tuple_adr            # address of text in text_data             (ntuple x 1)
        int*      tuple_size           # number of objects in tuple               (ntuple x 1)
        int*      tuple_objtype        # array of object types in all tuples      (ntupledata x 1)
        int*      tuple_objid          # array of object ids in all tuples        (ntupledata x 1)
        mjtNum*   tuple_objprm         # array of object params in all tuples     (ntupledata x 1)

        # keyframes
        mjtNum*   key_time             # key time                                 (nkey x 1)
        mjtNum*   key_qpos             # key position                             (nkey x nq)
        mjtNum*   key_qvel             # key velocity                             (nkey x nv)
        mjtNum*   key_act              # key activation                           (nkey x na)
        mjtNum*   key_mpos             # key mocap position                       (nkey x 3*nmocap)
        mjtNum*   key_mquat            # key mocap quaternion                     (nkey x 4*nmocap)
        mjtNum*   key_ctrl             # key control                              (nkey x nu)

        # names
        int*      name_bodyadr         # body name pointers                       (nbody x 1)
        int*      name_jntadr          # joint name pointers                      (njnt x 1)
        int*      name_geomadr         # geom name pointers                       (ngeom x 1)
        int*      name_siteadr         # site name pointers                       (nsite x 1)
        int*      name_camadr          # camera name pointers                     (ncam x 1)
        int*      name_lightadr        # light name pointers                      (nlight x 1)
        int*      name_flexadr         # flex name pointers                       (nflex x 1)
        int*      name_meshadr         # mesh name pointers                       (nmesh x 1)
        int*      name_skinadr         # skin name pointers                       (nskin x 1)
        int*      name_hfieldadr       # hfield name pointers                     (nhfield x 1)
        int*      name_texadr          # texture name pointers                    (ntex x 1)
        int*      name_matadr          # material name pointers                   (nmat x 1)
        int*      name_pairadr         # geom pair name pointers                  (npair x 1)
        int*      name_excludeadr      # exclude name pointers                    (nexclude x 1)
        int*      name_eqadr           # equality constraint name pointers        (neq x 1)
        int*      name_tendonadr       # tendon name pointers                     (ntendon x 1)
        int*      name_actuatoradr     # actuator name pointers                   (nu x 1)
        int*      name_sensoradr       # sensor name pointers                     (nsensor x 1)
        int*      name_numericadr      # numeric name pointers                    (nnumeric x 1)
        int*      name_textadr         # text name pointers                       (ntext x 1)
        int*      name_tupleadr        # tuple name pointers                      (ntuple x 1)
        int*      name_keyadr          # keyframe name pointers                   (nkey x 1)
        int*      name_pluginadr       # plugin instance name pointers            (nplugin x 1)
        char*     names                # names of all objects, 0-terminated       (nnames x 1)
        int*      names_map            # internal hash map of names               (nnames_map x 1)

        # paths
        char*     paths                # paths to assets, 0-terminated            (npaths x 1)


cdef extern from "mujoco/mjthread.h" nogil:
    enum: mjMAXTHREAD


cdef extern from "<stdint.h>" nogil:
    ctypedef  size_t uintptr_t


cdef extern from "mujoco/mjdata.h" nogil:

    enum: mjSTATE_INTEGRATION
    enum: mjNWARNING
    enum: mjNTIMER
    ctypedef struct mjVFS:
        pass

    ctypedef struct mjContact:                # result of collision detection functions
        # contact parameters set by near-phase collision function
        mjtNum  dist                    # distance between nearest points neg: penetration
        mjtNum  pos[3]                  # position of contact point: midpoint between geoms
        mjtNum  frame[9]                # normal is in [0-2], points from geom[0] to geom[1]

        # contact parameters set by mj_collideGeoms
        mjtNum  includemargin           # include if dist<includemargin=margin-gap
        mjtNum  friction[5]             # tangent1, 2, spin, roll1, 2
        mjtNum  solref[mjNREF]          # constraint solver reference, normal direction
        mjtNum  solreffriction[mjNREF]  # constraint solver reference, friction directions
        mjtNum  solimp[mjNIMP]          # constraint solver impedance

        # internal storage used by solver
        mjtNum  mu                      # friction of regularized cone, set by mj_makeConstraint
        mjtNum  H[36]                   # cone Hessian, set by mj_updateConstraint

        # contact descriptors set by mj_collideXXX
        int     dim                     # contact space dimensionality: 1, 3, 4 or 6
        int     geom1                   # id of geom 1 deprecated, use geom[0]
        int     geom2                   # id of geom 2 deprecated, use geom[1]
        int     geom[2]                 # geom ids -1 for flex
        int     flex[2]                 # flex ids -1 for geom
        int     elem[2]                 # element ids -1 for geom or flex vertex
        int     vert[2]                 # vertex ids  -1 for geom or flex element

        # flag set by mj_setContact or mj_instantiateContact
        int     exclude                 # 0: include, 1: in gap, 2: fused, 3: no dofs

        # address computed by mj_instantiateContact
        int     efc_address             # address in efc -1: not included

    #---------------------------------- diagnostics ---------------------------------------------------

    ctypedef struct mjWarningStat:      # warning statistics
        int     lastinfo          # info from last warning
        int     number            # how many times was warning raised


    ctypedef struct mjTimerStat:        # timer statistics
        mjtNum  duration          # cumulative duration
        int     number            # how many times was timer called


    ctypedef struct mjSolverStat:       # per-iteration solver statistics
        mjtNum  improvement       # cost reduction, scaled by 1/trace(M(qpos0))
        mjtNum  gradient          # gradient norm (primal only, scaled)
        mjtNum  lineslope         # slope in linesearch
        int     nactive           # number of active constraints
        int     nchange           # number of constraint state changes
        int     neval             # number of cost evaluations in line search
        int     nupdate           # number of Cholesky updates in line search

    #---------------------------------- mjData --------------------------------------------------------

    ctypedef struct mjData:
        # constant sizes
        size_t  narena            # size of the arena in bytes (inclusive of the stack)
        size_t  nbuffer           # size of main buffer in bytes
        int     nplugin           # number of plugin instances

        # stack pointer
        size_t  pstack            # first available mjtNum address in stack
        size_t  pbase             # value of pstack when mj_markStack was last called

        # arena pointer
        size_t  parena            # first available byte in arena

        # memory utilization stats
        size_t  maxuse_stack                      # maximum stack allocation in bytes
        size_t  maxuse_threadstack[mjMAXTHREAD]   # maximum stack allocation per thread in bytes
        size_t  maxuse_arena                      # maximum arena allocation in bytes
        int     maxuse_con                        # maximum number of contacts
        int     maxuse_efc                        # maximum number of scalar constraints

        # diagnostics
        mjWarningStat warning[mjNWARNING]  # warning statistics
        mjTimerStat   timer[mjNTIMER]      # timer statistics

        # solver statistics
        mjSolverStat  solver[mjNISLAND*mjNSOLVER]  # solver statistics per island, per iteration
        int     solver_nisland           # number of islands processed by solver
        int     solver_niter[mjNISLAND]  # number of solver iterations, per island
        int     solver_nnz[mjNISLAND]    # number of non-zeros in Hessian or efc_AR, per island
        mjtNum  solver_fwdinv[2]         # forward-inverse comparison: qfrc, efc

        # variable sizes
        int     ne                # number of equality constraints
        int     nf                # number of friction constraints
        int     nl                # number of limit constraints
        int     nefc              # number of constraints
        int     nnzJ              # number of non-zeros in constraint Jacobian
        int     ncon              # number of detected contacts
        int     nisland           # number of detected constraint islands

        # global properties
        mjtNum  time              # simulation time
        mjtNum  energy[2]         # potential, kinetic energy

        #-------------------- end of info header

        # buffers
        void*   buffer            # main buffer all pointers point in it                (nbuffer bytes)
        void*   arena             # arena+stack buffer                     (nstack*sizeof(mjtNum) bytes)

        #-------------------- main inputs and outputs of the computation

        # state
        mjtNum* qpos              # position                                         (nq x 1)
        mjtNum* qvel              # velocity                                         (nv x 1)
        mjtNum* act               # actuator activation                              (na x 1)
        mjtNum* qacc_warmstart    # acceleration used for warmstart                  (nv x 1)
        mjtNum* plugin_state      # plugin state                                     (npluginstate x 1)

        # control
        mjtNum* ctrl              # control                                          (nu x 1)
        mjtNum* qfrc_applied      # applied generalized force                        (nv x 1)
        mjtNum* xfrc_applied      # applied Cartesian force/torque                   (nbody x 6)
        mjtByte* eq_active        # enable/disable constraints                       (neq x 1)

        # mocap data
        mjtNum* mocap_pos         # positions of mocap bodies                        (nmocap x 3)
        mjtNum* mocap_quat        # orientations of mocap bodies                     (nmocap x 4)

        # dynamics
        mjtNum* qacc              # acceleration                                     (nv x 1)
        mjtNum* act_dot           # time-derivative of actuator activation           (na x 1)

        # user data
        mjtNum* userdata          # user data, not touched by engine                 (nuserdata x 1)

        # sensors
        mjtNum* sensordata        # sensor data array                                (nsensordata x 1)

        # plugins
        int*       plugin         # copy of m->plugin, required for deletion         (nplugin x 1)
        uintptr_t* plugin_data    # pointer to plugin-managed data structure         (nplugin x 1)

        #-------------------- POSITION dependent

        # computed by mj_fwdPosition/mj_kinematics
        mjtNum* xpos              # Cartesian position of body frame                 (nbody x 3)
        mjtNum* xquat             # Cartesian orientation of body frame              (nbody x 4)
        mjtNum* xmat              # Cartesian orientation of body frame              (nbody x 9)
        mjtNum* xipos             # Cartesian position of body com                   (nbody x 3)
        mjtNum* ximat             # Cartesian orientation of body inertia            (nbody x 9)
        mjtNum* xanchor           # Cartesian position of joint anchor               (njnt x 3)
        mjtNum* xaxis             # Cartesian joint axis                             (njnt x 3)
        mjtNum* geom_xpos         # Cartesian geom position                          (ngeom x 3)
        mjtNum* geom_xmat         # Cartesian geom orientation                       (ngeom x 9)
        mjtNum* site_xpos         # Cartesian site position                          (nsite x 3)
        mjtNum* site_xmat         # Cartesian site orientation                       (nsite x 9)
        mjtNum* cam_xpos          # Cartesian camera position                        (ncam x 3)
        mjtNum* cam_xmat          # Cartesian camera orientation                     (ncam x 9)
        mjtNum* light_xpos        # Cartesian light position                         (nlight x 3)
        mjtNum* light_xdir        # Cartesian light direction                        (nlight x 3)

        # computed by mj_fwdPosition/mj_comPos
        mjtNum* subtree_com       # center of mass of each subtree                   (nbody x 3)
        mjtNum* cdof              # com-based motion axis of each dof (rot:lin)      (nv x 6)
        mjtNum* cinert            # com-based body inertia and mass                  (nbody x 10)

        # computed by mj_fwdPosition/mj_flex
        mjtNum* flexvert_xpos     # Cartesian flex vertex positions                  (nflexvert x 3)
        mjtNum* flexelem_aabb     # flex element bounding boxes (center, size)       (nflexelem x 6)
        int*    flexedge_J_rownnz # number of non-zeros in Jacobian row              (nflexedge x 1)
        int*    flexedge_J_rowadr # row start address in colind array                (nflexedge x 1)
        int*    flexedge_J_colind # column indices in sparse Jacobian                (nflexedge x nv)
        mjtNum* flexedge_J        # flex edge Jacobian                               (nflexedge x nv)
        mjtNum* flexedge_length   # flex edge lengths                                (nflexedge x 1)

        # computed by mj_fwdPosition/mj_tendon
        int*    ten_wrapadr       # start address of tendon's path                   (ntendon x 1)
        int*    ten_wrapnum       # number of wrap points in path                    (ntendon x 1)
        int*    ten_J_rownnz      # number of non-zeros in Jacobian row              (ntendon x 1)
        int*    ten_J_rowadr      # row start address in colind array                (ntendon x 1)
        int*    ten_J_colind      # column indices in sparse Jacobian                (ntendon x nv)
        mjtNum* ten_J             # tendon Jacobian                                  (ntendon x nv)
        mjtNum* ten_length        # tendon lengths                                   (ntendon x 1)
        int*    wrap_obj          # geom id -1: site -2: pulley                    (nwrap*2 x 1)
        mjtNum* wrap_xpos         # Cartesian 3D points in all path                  (nwrap*2 x 3)

        # computed by mj_fwdPosition/mj_transmission
        mjtNum* actuator_length   # actuator lengths                                 (nu x 1)
        mjtNum* actuator_moment   # actuator moments                                 (nu x nv)

        # computed by mj_fwdPosition/mj_crb
        mjtNum* crb               # com-based composite inertia and mass             (nbody x 10)
        mjtNum* qM                # total inertia (sparse)                           (nM x 1)

        # computed by mj_fwdPosition/mj_factorM
        mjtNum* qLD               # L'*D*L factorization of M (sparse)               (nM x 1)
        mjtNum* qLDiagInv         # 1/diag(D)                                        (nv x 1)
        mjtNum* qLDiagSqrtInv     # 1/sqrt(diag(D))                                  (nv x 1)

        # computed by mj_collisionTree
        mjtNum*  bvh_aabb_dyn     # global bounding box (center, size)               (nbvhdynamic x 6)
        mjtByte* bvh_active       # volume has been added to collisions              (nbvh x 1)

        #-------------------- POSITION, VELOCITY dependent

        # computed by mj_fwdVelocity
        mjtNum* flexedge_velocity # flex edge velocities                             (nflexedge x 1)
        mjtNum* ten_velocity      # tendon velocities                                (ntendon x 1)
        mjtNum* actuator_velocity # actuator velocities                              (nu x 1)

        # computed by mj_fwdVelocity/mj_comVel
        mjtNum* cvel              # com-based velocity (rot:lin)                     (nbody x 6)
        mjtNum* cdof_dot          # time-derivative of cdof (rot:lin)                (nv x 6)

        # computed by mj_fwdVelocity/mj_rne (without acceleration)
        mjtNum* qfrc_bias         # C(qpos,qvel)                                     (nv x 1)

        # computed by mj_fwdVelocity/mj_passive
        mjtNum* qfrc_spring       # passive spring force                             (nv x 1)
        mjtNum* qfrc_damper       # passive damper force                             (nv x 1)
        mjtNum* qfrc_gravcomp     # passive gravity compensation force               (nv x 1)
        mjtNum* qfrc_fluid        # passive fluid force                              (nv x 1)
        mjtNum* qfrc_passive      # total passive force                              (nv x 1)

        # computed by mj_sensorVel/mj_subtreeVel if needed
        mjtNum* subtree_linvel    # linear velocity of subtree com                   (nbody x 3)
        mjtNum* subtree_angmom    # angular momentum about subtree com               (nbody x 3)

        # computed by mj_Euler or mj_implicit
        mjtNum* qH                # L'*D*L factorization of modified M               (nM x 1)
        mjtNum* qHDiagInv         # 1/diag(D) of modified M                          (nv x 1)

        # computed by mj_resetData
        int*    D_rownnz          # non-zeros in each row                            (nv x 1)
        int*    D_rowadr          # address of each row in D_colind                  (nv x 1)
        int*    D_colind          # column indices of non-zeros                      (nD x 1)
        int*    B_rownnz          # non-zeros in each row                            (nbody x 1)
        int*    B_rowadr          # address of each row in B_colind                  (nbody x 1)
        int*    B_colind          # column indices of non-zeros                      (nB x 1)

        # computed by mj_implicit/mj_derivative
        mjtNum* qDeriv            # d (passive + actuator - bias) / d qvel           (nD x 1)

        # computed by mj_implicit/mju_factorLUSparse
        mjtNum* qLU               # sparse LU of (qM - dt*qDeriv)                    (nD x 1)

        #-------------------- POSITION, VELOCITY, CONTROL/ACCELERATION dependent

        # computed by mj_fwdActuation
        mjtNum* actuator_force    # actuator force in actuation space                (nu x 1)
        mjtNum* qfrc_actuator     # actuator force                                   (nv x 1)

        # computed by mj_fwdAcceleration
        mjtNum* qfrc_smooth       # net unconstrained force                          (nv x 1)
        mjtNum* qacc_smooth       # unconstrained acceleration                       (nv x 1)

        # computed by mj_fwdConstraint/mj_inverse
        mjtNum* qfrc_constraint   # constraint force                                 (nv x 1)

        # computed by mj_inverse
        mjtNum* qfrc_inverse      # net external force should equal:                (nv x 1)
                                 # qfrc_applied + J'*xfrc_applied + qfrc_actuator

        # computed by mj_sensorAcc/mj_rnePostConstraint if needed rotation:translation format
        mjtNum* cacc              # com-based acceleration                           (nbody x 6)
        mjtNum* cfrc_int          # com-based interaction force with parent          (nbody x 6)
        mjtNum* cfrc_ext          # com-based external force on body                 (nbody x 6)

        #-------------------- arena-allocated: POSITION dependent

        # computed by mj_collision
        mjContact* contact        # list of all detected contacts                    (ncon x 1)

        # computed by mj_makeConstraint
        int*    efc_type          # constraint type (mjtConstraint)                  (nefc x 1)
        int*    efc_id            # id of object of specified type                   (nefc x 1)
        int*    efc_J_rownnz      # number of non-zeros in constraint Jacobian row   (nefc x 1)
        int*    efc_J_rowadr      # row start address in colind array                (nefc x 1)
        int*    efc_J_rowsuper    # number of subsequent rows in supernode           (nefc x 1)
        int*    efc_J_colind      # column indices in constraint Jacobian            (nnzJ x 1)
        int*    efc_JT_rownnz     # number of non-zeros in constraint Jacobian row T (nv x 1)
        int*    efc_JT_rowadr     # row start address in colind array              T (nv x 1)
        int*    efc_JT_rowsuper   # number of subsequent rows in supernode         T (nv x 1)
        int*    efc_JT_colind     # column indices in constraint Jacobian          T (nnzJ x 1)
        mjtNum* efc_J             # constraint Jacobian                              (nnzJ x 1)
        mjtNum* efc_JT            # constraint Jacobian transposed                   (nnzJ x 1)
        mjtNum* efc_pos           # constraint position (equality, contact)          (nefc x 1)
        mjtNum* efc_margin        # inclusion margin (contact)                       (nefc x 1)
        mjtNum* efc_frictionloss  # frictionloss (friction)                          (nefc x 1)
        mjtNum* efc_diagApprox    # approximation to diagonal of A                   (nefc x 1)
        mjtNum* efc_KBIP          # stiffness, damping, impedance, imp'              (nefc x 4)
        mjtNum* efc_D             # constraint mass                                  (nefc x 1)
        mjtNum* efc_R             # inverse constraint mass                          (nefc x 1)
        int*    tendon_efcadr     # first efc address involving tendon -1: none     (ntendon x 1)

        # computed by mj_island
        int*    dof_island        # island id of this dof -1: none                  (nv x 1)
        int*    island_dofnum     # number of dofs in island                         (nisland x 1)
        int*    island_dofadr     # start address in island_dofind                   (nisland x 1)
        int*    island_dofind     # island dof indices -1: none                     (nv x 1)
        int*    dof_islandind     # dof island indices -1: none                     (nv x 1)
        int*    efc_island        # island id of this constraint                     (nefc x 1)
        int*    island_efcnum     # number of constraints in island                  (nisland x 1)
        int*    island_efcadr     # start address in island_efcind                   (nisland x 1)
        int*    island_efcind     # island constraint indices                        (nefc x 1)

        # computed by mj_projectConstraint (dual solver)
        int*    efc_AR_rownnz     # number of non-zeros in AR                        (nefc x 1)
        int*    efc_AR_rowadr     # row start address in colind array                (nefc x 1)
        int*    efc_AR_colind     # column indices in sparse AR                      (nefc x nefc)
        mjtNum* efc_AR            # J*inv(M)*J' + R                                  (nefc x nefc)

        #-------------------- arena-allocated: POSITION, VELOCITY dependent

        # computed by mj_fwdVelocity/mj_referenceConstraint
        mjtNum* efc_vel           # velocity in constraint space: J*qvel             (nefc x 1)
        mjtNum* efc_aref          # reference pseudo-acceleration                    (nefc x 1)

        #-------------------- arena-allocated: POSITION, VELOCITY, CONTROL/ACCELERATION dependent

        # computed by mj_fwdConstraint/mj_inverse
        mjtNum* efc_b            # linear cost term: J*qacc_smooth - aref            (nefc x 1)
        mjtNum* efc_force        # constraint force in constraint space              (nefc x 1)
        int*    efc_state        # constraint state (mjtConstraintState)             (nefc x 1)

        # ThreadPool for multithreaded operations
        uintptr_t threadpool


cdef extern from "mujoco/mujoco.h" nogil:

    #---------------------------------- Parse and compile ---------------------------------------------

    # Parse XML file in MJCF or URDF format, compile it, return low-level model.
    # If vfs is not NULL, look up files in vfs before reading from disk.
    # If error is not NULL, it must have size error_sz.
    mjModel* mj_loadXML(const char* filename, const mjVFS* vfs, char* error, int error_sz)

    # Update XML data structures with info from low-level model, save as MJCF.
    # If error is not NULL, it must have size error_sz.
    #int mj_saveLastXML(const char* filename, const mjModel* m, char* error, int error_sz)

    # Free last XML model if loaded. Called internally at each load.
    #void mj_freeLastXML(void)

    # Print internal XML schema as plain text or HTML, with style-padding or &nbsp.
    #int mj_printSchema(const char* filename, char* buffer, int buffer_sz,
    #                         int flg_html, int flg_pad)


    #---------------------------------- Main simulation -----------------------------------------------

    # Advance simulation, use control callback to obtain external force and control.
    void mj_step(const mjModel* m, mjData* d)

    # Advance simulation in two steps: before external force and control is set by user.
    void mj_step1(const mjModel* m, mjData* d)

    # Advance simulation in two steps: after external force and control is set by user.
    void mj_step2(const mjModel* m, mjData* d)

    # Forward dynamics: same as mj_step but do not integrate in time.
    void mj_forward(const mjModel* m, mjData* d)

    # Inverse dynamics: qacc must be set before calling.
    void mj_inverse(const mjModel* m, mjData* d)

    # Forward dynamics with skip skipstage is mjtStage.
    void mj_forwardSkip(const mjModel* m, mjData* d, int skipstage, int skipsensor)

    # Inverse dynamics with skip skipstage is mjtStage.
    void mj_inverseSkip(const mjModel* m, mjData* d, int skipstage, int skipsensor)


    #---------------------------------- Initialization ------------------------------------------------
    # Copy mjModel, allocate new if dest is NULL.
    mjModel* mj_copyModel(mjModel* dest, const mjModel* src)

    # Copy mjModel, allocate new if dest is NULL.
    mjModel* mj_copyModel(mjModel* dest, const mjModel* src)

    # Load model from binary MJB file.
    # If vfs is not NULL, look up file in vfs before reading from disk.
    mjModel* mj_loadModel(const char* filename, const mjVFS* vfs)

    # Free memory allocation in model.
    void mj_deleteModel(mjModel* m)

    # Allocate mjData corresponding to given model.
    # If the model buffer is unallocated the initial configuration will not be set.
    mjData* mj_makeData(const mjModel* m)

    # Copy mjData.
    # m is only required to contain the size fields from MJMODEL_INTS.
    mjData* mj_copyData(mjData* dest, const mjModel* m, const mjData* src)

    # Reset data to defaults.
    void mj_resetData(const mjModel* m, mjData* d)

    # Free memory allocation in mjData.
    void mj_deleteData(mjData* d)

    #---------------------------------- Support -------------------------------------------------------

    # Return size of state specification.
    int mj_stateSize(const mjModel* m, unsigned int spec)

    # Get state.
    void mj_getState(const mjModel* m, const mjData* d, mjtNum* state, unsigned int spec)

    # Set state.
    void mj_setState(const mjModel* m, mjData* d, const mjtNum* state, unsigned int spec)
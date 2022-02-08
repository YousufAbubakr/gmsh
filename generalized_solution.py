# ------------------------------------------------------------------------------
#
#  Gmsh Python API
#
#  The Generalized Mesh Solution to the 3-Layer Tissue AF Model
#
# ------------------------------------------------------------------------------

import gmsh
import sys
from sympy import cos, sin, pi, tan, sec

gmsh.initialize(sys.argv)
gmsh.model.add("solution")

#This model produces a mesh based off the CAD model "matrix_split_SOP.SLDPRT" 
#and "fibers_SOP.SLDPRT," which represents one layer of the 3-Layer Tissue AF
#model. This CAD model splits up each layer into several, individual matrix-fiber
#bodies. In this meshing approach, the function, "matrixfibergeo", is defined 
#to setup the geometry (points, lines, surfaces, and volumes), while the
#function, "matrixfibermesh", defines the transfinite settings. The geometry setup
#is based entirely on the CAD model, so using these specific SolidWorks Part files
#as a reference is going to be very helpful when reading comment statements.

#All distances are in units of mm

#Some key things to note for the transfinite settings:
    #Transfinite Surfaces can have as many boundary nodes as possible, but if
    #the surface has MORE than 4 boundary nodes, they must be explicity stated 
    #in the transfinite surface function calls
    #Transfinite volumes MUST have 6 OR 8 boundary nodes and can be stated
    #explicitly in the transfinite volume caling process

#To simplify the function-calling process, gmshOCC and gmshMOD is defined:
gmshOCC = gmsh.model.occ
gmshMOD = gmsh.model
gmshMESH = gmsh.model.mesh

#Utility Functions:    
def dimTagReturner(dim, dimTags):
    """
     dimTagReturner(dim, dimTags)

    Returns a list of tuple dimTags in 'newDimTags' such that only the dimTags in
    'dimTags' with dimension 'dim' are kept

    Returns list of dimTags.
    """
    newDimTags = [elem for elem in dimTags if elem[0] == dim]
    return newDimTags

def matrixPrismGenertor(centers, xtranslations, ytranslations, ztranslations):
    """
     matrixPrismGenertor(centers, xtranslations, ytranslations, ztranslations)

    Function that generates AND connects extruded curve enities for specific anti-parallel
    matrix geometry in the CAD model. Function is under the assumption that desired
    matrix geometry is a rectangular prism with centers on the source and target
    face described in the list given by 'centers' and the translations that correspond
    to the source face are in 'xtranslation' and the first element in 'ytranslations,'
    and the translations that correspond to the target face are in 'ztranslations' and
    the second element of 'ztranslations.'

    Returns list of extruded curve entities.
    """
    assert len(centers) == len(ytranslations), "Centers list must be the same length as y-translations list."
    extrusions = []
    #Defining the source point and creating extrusions to create base for source face:
    center1_start = gmshOCC.addPoint(centers[0][0], centers[0][1], centers[0][2])
    extrusions.append(gmshOCC.extrude([(0, center1_start)], 0, ytranslations[0], 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), -xtranslations[0], 0, 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), -xtranslations[1], 0, 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, -ytranslations[0], 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, -ytranslations[0], 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), xtranslations[1], 0, 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), xtranslations[0], 0, 0))
    extrusions.append(tuple([1, gmshOCC.addLine(gmshOCC.getMaxTag(0), center1_start)]))
    #Defining the target point and creating extrusions to create base for target face:
    center2_start = gmshOCC.addPoint(centers[1][0], centers[1][1], centers[1][2])
    extrusions.append(gmshOCC.extrude([(0, center2_start)], 0, ytranslations[1], 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, -ztranslations[0]))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, -ztranslations[1]))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, -ytranslations[1], 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, -ytranslations[1], 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, ztranslations[1]))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, ztranslations[0]))
    extrusions.append(tuple([1, gmshOCC.addLine(gmshOCC.getMaxTag(0), center2_start)]))
    #Connecting the source and target base faces with lines:
    for i in range(8):
        extrusions.append(gmshOCC.addLine(center1_start + i, center2_start + i))
    return extrusions

def curveLoopGenerator(curveTags, loopBools, iterNums, jumpBools, curveJump = 1, pointTags = []):
    """
     curveLoopGenerator(curveTags, loopBools, iterNums, jumpBools, curveJump = 1, pointTags = [])

    Generated curve loop and surface entities based off of the curve loop pattern
    given by the model's geometry setup. Usually, the curves are oriented CCW, so 
    this function is designed to create surfaces based off of this direction.
    'curveTags' are associated curve tags that relate to this CCW orientation, and
    'loopBools' is a boolean list of the same length as 'curveTags' that says
    whether or not the curve tag of the same element position in 'curveTags' will
    have a circular iteration condition assoiated with its curve loop definition
    up to a certain number of iterations, 'iterNum'

    Returns list of surfaces being defined.
    """
    assert len(curveTags) == len(loopBools), "Curve tags list must be the same length as loop booleans list."
    curveTagsLst = []
    if curveJump == 1:
        for i in range(len(curveTags)):
            if loopBools[i]:
                curveTagsLst.append([curveTags[i] + j if j < iterNums - 1 else curveTags[i] - 1 for j in range(iterNums)])
            else:
                curveTagsLst.append([curveTags[i] + j for j in range(iterNums)])
    else:
        for i in range(len(curveTags)):
            if jumpBools[i]:
                if loopBools[i]:
                    curveTagsLst.append([curveTags[i] + j*curveJump if j < iterNums - 1 else curveTags[i] - 1 for j in range(iterNums)])
                else:
                    curveTagsLst.append([curveTags[i] + j*curveJump for j in range(iterNums)])
            else:
                if loopBools[i]:
                    curveTagsLst.append([curveTags[i] + j if j < iterNums - 1 else curveTags[i] - 1 for j in range(iterNums)])
                else:
                    curveTagsLst.append([curveTags[i] + j for j in range(iterNums)])
    curveLoopZipped = zip(*curveTagsLst)
    curveLoopLst = list(curveLoopZipped)
    surfLoopLst = []
    iter = 0
    for elem in curveLoopLst:
        curves = gmshOCC.addCurveLoop(list(elem))
        if pointTags == []:
            surfLoopLst.append(gmshOCC.addSurfaceFilling(curves))
        else:
            surfLoopLst.append(gmshOCC.addSurfaceFilling(curves, -1, [pointTags[0] + iter, pointTags[1] + iter]))
            iter += 2
    #print("Curve Loops being defined: ", curveLoopLst)
    return surfLoopLst

def surfaceLoopGenerator(surfTags, loopBools, iterNums, jumpBools, surfJump = 1, sewBool = True):
    """
     surfaceLoopGenerator(surfTags, loopBools, iterNums, jumpBools, surfJump = 1)

    Generated curve loop and surface entities based off of the curve loop pattern
    given by the model's geometry setup. Usually, the curves are oriented CCW, so 
    this function is designed to create surfaces based off of this direction.
    'curveTags' are associated curve tags that relate to this CCW orientation, and
    'loopBools' is a boolean list of the same length as 'curveTags' that says
    whether or not the curve tag of the same element position in 'curveTags' will
    have a circular iteration condition assoiated with its curve loop definition
    up to a certain number of iterations, 'iterNum'

    Returns list of curve loops being defined.
    """
    assert len(surfTags) == len(loopBools), "Curve tags list must be the same length as loop booleans list."
    surfTagsLst = []
    if surfJump == 1:
        for i in range(len(surfTags)):
            if loopBools[i]:
                surfTagsLst.append([surfTags[i] + j if j < iterNums - 1 else surfTags[i] - 1 for j in range(iterNums)])
            else:
                surfTagsLst.append([surfTags[i] + j for j in range(iterNums)])
    else:
        for i in range(len(surfTags)):
            if jumpBools[i]:
                if loopBools[i]:
                    surfTagsLst.append([surfTags[i] + j*surfJump if j < iterNums - 1 else surfTags[i] - 1 for j in range(iterNums)])
                else:
                    surfTagsLst.append([surfTags[i] + j*surfJump for j in range(iterNums)])
            else:
                if loopBools[i]:
                    surfTagsLst.append([surfTags[i] + j if j < iterNums - 1 else surfTags[i] - 1 for j in range(iterNums)])
                else:
                    surfTagsLst.append([surfTags[i] + j for j in range(iterNums)])
    surfLoopZipped = zip(*surfTagsLst)
    surfLoopLst = list(surfLoopZipped)
    volLst = []
    for elem in surfLoopLst:
        #print(list(elem))
        surfs = gmshOCC.addSurfaceLoop(list(elem), -1, sewBool)
        volLst.append(gmshOCC.addVolume([surfs]))
    #print("Surface Loops being defined: ", surfLoopLst)
    return volLst

#Main Functions:
def matrixfibergeo_right(matrix_ID, transfinite_curves, angle = 30, but1 = 1/3, but2 = 2/9):
    """
     matrixfibergeo_right(matrix_ID, transfinite_curves, angle = 30, but1 = 1/3, but2 = 2/9)

    Defines geometry for one matrix-fiber set *on the right hand side of the entire layer* 
    that is based off of the SolidWorks Part files, starting with the matrix-fiber 
    set with fibers angled cross-ply at some angle 'angle' with matrix ID, 'matrix_ID', 
    such that a matrix ID of 0 corresponds to the right-most, corner-most matrix-fiber 
    body, and 'but1' and 'but2' represent the butterfly ratios, which is the ratio 
    determining how large the butterfly prism will be with respect to the elliptical 
    boundary, for the source and target butterfly prisms. 'transfinite_curves' is 
    a list of dim = 2 that contains numNodes data, such that the first element 
    represents the number of nodes on the source and target face curves, while 
    the second element represents the number of nodes on the curves in between 
    the source and target faces.

    Returns matrix-fiber set geometry entities *on the right hand side*
    """
    #Main Dimensions of one Matrix-Fiber Layer:
    height = 11
    length = 6
    cent = 0.22  #fiber centerline distance
    d = 0.12  #fiber diameter
    r = d/2  #fiber radius
    t = 0.2  #matrix thickness
    ang = angle  #fiber orientation in deg

    #The width of one of these matrix bodies projected onto the most
    #bottom-most edge in the geometry in mm:
    width = cent/cos(ang*pi/180)

    #The height of one of these matrix bodies projected onto the most
    #right-most edge in the geometry in mm:
    height = cent/sin(ang*pi/180)

    #Corner offset, which avoids weird, unsymmetrical fiber geometries:
    off = 0.08

    #Number of fibers to the right of this corner offset matrix:
    N_fib = 22

    #The corner fibers in this model are always going to be weird and are edge cases
    #when defining a mesh for this geometry. Initially, it will be easier to ignore
    #these cases and start meshing the full matrix-fiber bodies themselves. Based off
    #of the SolidWorks parts, the origin (0, 0, 0) is defined as the bottom right hand
    #corner with the non-matrix-fiber body and edge with the 0.08 mm offset.

    #In dealing with this non-matrix-fiber body offset, the closest matrix body, which
    #is to the left of the defined origin (negative x-value), has some starting x
    #value for its right-most edge. After doing some trigonometry and geometry
    #breakdowns, this offset is defined as:
    start = length - width/2 - off - N_fib*width

    #Additionally, for geometry/mesh preparation, it will useful to identify these
    #matrix-fibers bodies by unique IDs. Starting off with the first offset matrix:
    mat_ID = matrix_ID
    
    #To simplify the function-calling process, gmshOCC and gmshMOD is defined:
    gmshOCC = gmsh.model.occ
    gmshMOD = gmsh.model

    #Geometry Generation
    #The origin:
    #origin = gmshOCC.addPoint(0, 0, 0)
    
    #Next, we wanna create the outer most boundaries of corner-most fiber. To do so,
    #we first define the source and target faces by starting at on of their midpoints.
    #This procedure will make more sense later on, when we define ellipse points.
    #Some key variables include:
    x_start = -start - mat_ID*width
    y_start = t/2
    z_start = -abs(x_start)*tan((90 - ang)*pi/180)
    #Next, we wull define the curve boundaries. To do so, we will primarily use the utility
    #function 'matrixPrismGenerator,' which fully defines the curves we would like
    #to make:
    if mat_ID == 22:
        matrix = matrixPrismGenertor([[x_start, y_start, 0], [0, y_start, z_start]], 
                                     [width/2, off], [t/2, t/2], [height/2, height/2])
    else:
        matrix = matrixPrismGenertor([[x_start, y_start, 0], [0, y_start, z_start]], 
                                     [width/2, width/2], [t/2, t/2], [height/2, height/2])
    #Defining relevant curve loop variables (will become apparent later on):
    matbound_curveloop = [matrix[0][1][1], matrix[16]]
    #Geometries with multiple matrix bodies will have interseting points and lines
    #at the points/surface of contact, but because overlapping entities don't
    #impact the mesh, the overlapping entities will stay in the model for
    #future model simplifications.
    
    #Next, we wanna create the geometry for the fiber volumes, which are really
    #elliptical cutouts in the matrix geometry. We start off be defining the
    #ellipse centers on each of the matrix source and target faces:
    x_ell_start = x_start - width/2
    y_ell_start = t/2
    z_ell_start = z_start - height/2
    maj_axis_source = r*sec(ang*pi/180)  #ellipse dimensions
    maj_axis_target = r*sec((90 - ang)*pi/180)
    min_axis = r
    center_source = [x_ell_start, y_ell_start, 0]
    center_target = [0, y_ell_start, z_ell_start]
    #Additionally, we will defining a relevant point variable (will become apparent later on):
    point_ref_tag = gmshOCC.addPoint(center_source[0], center_source[1], center_source[2])
    #For mat_ID = 0, point_ref_tag = 17
    gmshOCC.addPoint(center_target[0], center_target[1], center_target[2])
    #Next, we define source_ell and target_ell that stores the generated ellipse
    #curve data to be able to generate coneecting lines between source and
    #target face ellipses
    source_ell = []
    target_ell = []
    for i in range(8):
        source_ell.append(gmshOCC.addEllipse(x_ell_start, y_ell_start, 0, maj_axis_source, 
                                             min_axis, -1, i*pi/4, (i + 1)*pi/4))
        #print("Ellipse arcs on source face: ", source_ell, "Ellipse arcs on target face: ", target_ell)
    for i in range(8):
        target_ell.append(gmshOCC.addEllipse(0, y_ell_start, z_ell_start, maj_axis_target, 
                                             min_axis, -1, i*pi/4, (i + 1)*pi/4))
        #print("Ellipse arcs on source face: ", source_ell, "Ellipse arcs on target face: ", target_ell)
        #Because the addEllipse function does not allow you to define the orientation
        #in which the ellipse is defined, we must automatically rotate its curves
        #to define it for our model:
        gmshOCC.rotate([(1, target_ell[-1])], 0, y_ell_start, z_ell_start, 0, -1, 0, pi/2)
    #Current gmsh model needs to be synchronized onto OpenCASCADE CAD representation
    #in order to use stored source_ell and target_ell data
    #NOTE, generally you want to be able to reduce the number of times you
    #synchronize your model because it generally uses a nontrivial amount of memory
    gmshOCC.synchronize()
    
    s_ellipse_boundary = []
    t_ellipse_boundary = []
    for source, target in zip(source_ell, target_ell):
        s_ellipse_boundary.append(gmshMOD.getBoundary([(1, source)]))
        t_ellipse_boundary.append(gmshMOD.getBoundary([(1, target)]))
        ellipse_source_boundary = gmshMOD.getBoundary([(1, source)])
        ellipse_target_boundary = gmshMOD.getBoundary([(1, target)])
        #Connecting the source and target base faces with lines:
        gmshOCC.addLine(ellipse_source_boundary[0][1], ellipse_target_boundary[0][1])
    
    #Next, we need to create our third, final primary structural component of our
    #mesh: the Butterfly Prism in the center of the ellipse structure. To do so,
    #we define some key geometric variables:
    butterfly_ratio_1 = but1
    butterfly_ratio_2 = but2
    square_source =  butterfly_ratio_1*maj_axis_source
    square_target =  butterfly_ratio_2*maj_axis_target
    #Next, we wull define the curve boundaries. To do so, we will primarily use the utility
    #function 'matrixPrismGenerator,' which fully defines the curves we wouls like
    #to make:
    extruded_entities = matrixPrismGenertor([[center_source[0] + square_source, center_source[1], 
                       center_source[2]], [center_target[0], center_target[1], 
                                           center_target[2] + square_target]], 
                                           [square_source, square_source], 
                                           [square_source, square_target], 
                                           [square_target, square_target])
    #Defining relevant curve loop variables (will become apparent later on):
    extbound_curveloop = [extruded_entities[0][1][1], extruded_entities[16]]

    gmshOCC.synchronize()
    #Now that we've created all of the main meshing bodies, we have to define the
    #lines that connect these bodies. To do, we have to do a lot of data sorting
    #with the curve entities to obtain the boundary points that define these curves.
    #First, we get specifically the dim = 1 extruded entitiy data from the previous
    #command defined in 'extruded_entities':
    extruded_curves = [dimTagReturner(1, elem) if type(elem) is list else elem for elem in extruded_entities]
    extruded_dimtags = []
    for elem in extruded_curves:
        if type(elem) is list:
            extruded_dimtags.append(elem[0])
        else:
            extruded_dimtags.append(elem)
    extruded_dimtags_source = extruded_dimtags[:8]
    extruded_dimtags_target = extruded_dimtags[8:len(extruded_dimtags)]
     
    #Synchronize model to access .model function 'getBoundary'
    gmshOCC.synchronize()
    
    #Using 'getBoundary' to obtain boundary points of curves defined in
    #'extruded_dimtags_source' and 'extruded_dimtags_target'
    s_extruded_boundary = []
    t_extruded_boundary = []
    for source, target in zip(extruded_dimtags_source, extruded_dimtags_target):
        s_extruded_boundary.append(gmshMOD.getBoundary([source]))
        t_extruded_boundary.append(gmshMOD.getBoundary([target]))
    #Putting everything together to draw out the boundary point tags to use to 
    #add connecting lines between butterfly prism and ellipse boundaries
    s_extruded_boundary[-1].reverse()
    for ellipse, extruded in zip(s_ellipse_boundary, s_extruded_boundary):
        ellipse_source_point = ellipse[0][1]
        extruded_source_point = extruded[0][1]
        gmshOCC.addLine(ellipse_source_point, extruded_source_point)
    t_extruded_boundary[-1].reverse()
    for ellipse, extruded in zip(t_ellipse_boundary, t_extruded_boundary):
        ellipse_target_point = ellipse[0][1]
        extruded_target_point = extruded[0][1]
        gmshOCC.addLine(ellipse_target_point, extruded_target_point)
    
    #Next, we get the corresponding curve entitity data from the main matrix body
    #and we do almost the same exact procedure:
    matrix_curves = [dimTagReturner(1, elem) if type(elem) is list else elem for elem in matrix]
    matrix_dimtags = []
    for elem in matrix_curves:
        if type(elem) is list:
            matrix_dimtags.append(elem[0])
        else:
            matrix_dimtags.append(elem)
    matrix_dimtags_source = matrix_dimtags[:8]
    matrix_dimtags_target = matrix_dimtags[8:len(extruded_dimtags)]

    gmshOCC.synchronize()
    
    #Using 'getBoundary' to obtain boundary points of curves defined in
    #'extruded_dimtags_source' and 'extruded_dimtags_target'
    s_matrix_boundary = []
    t_matrix_boundary = []
    for source, target in zip(matrix_dimtags_source, matrix_dimtags_target):
        s_matrix_boundary.append(gmshMOD.getBoundary([source]))
        t_matrix_boundary.append(gmshMOD.getBoundary([target]))
    #Putting everything together to draw out the boundary point tags to use to 
    #add connecting lines between butterfly prism and ellipse boundaries
    s_matrix_boundary[-1].reverse()
    for ellipse, matrix in zip(s_ellipse_boundary, s_matrix_boundary):
        ellipse_source_point = ellipse[0][1]
        matrix_source_point = matrix[0][1]
        gmshOCC.addLine(ellipse_source_point, matrix_source_point)
    t_matrix_boundary[-1].reverse()
    for ellipse, matrix in zip(t_ellipse_boundary, t_matrix_boundary):
        ellipse_target_point = ellipse[0][1]
        matrix_target_point = matrix[0][1]
        gmshOCC.addLine(ellipse_target_point, matrix_target_point)
    
    #Great, now we essentially have all of our points and curves fully defined!
    #Now, we actually have to use these points/curves to define curveloops, surfaces,
    #and volumes. Let's focus primarily on the outer matrix body first. To do so, 
    #we will primarily use the utility function 'curveLoopGenerator,' which fully
    #defines the surfaces we would like to make:
    #Outer matrix boundary:
    curveLoopGenerator([matbound_curveloop[0], matbound_curveloop[1], 
                        matbound_curveloop[0] + 8, matbound_curveloop[1] + 1], 
                           [False, False, False, True], 8, [False, False, False, False])
    #The butterfly prism, ellipses, and intermediate surfaces require a similar 
    #procedure as well:
    #Butterfly prism (8 individual surfaces):
    curveLoopGenerator([extbound_curveloop[0], extbound_curveloop[1], 
                        extbound_curveloop[0] + 8, extbound_curveloop[1] + 1], 
                            [False, False, False, True], 8, [False, False, False, False])
    #Butterfly prism (4 grouped surfaces):
    curves1 = gmshOCC.addCurveLoop([extbound_curveloop[0] + 7, extbound_curveloop[0], 
                                    extbound_curveloop[1] + 1, extbound_curveloop[0] + 8, 
                                    extbound_curveloop[0] + 15, extbound_curveloop[1] + 7])
    gmshOCC.addSurfaceFilling(curves1)
    curves2 = gmshOCC.addCurveLoop([extbound_curveloop[0] + 1, extbound_curveloop[0] + 2, 
                                    extbound_curveloop[1] + 3, extbound_curveloop[0] + 10, 
                                    extbound_curveloop[0] + 9, extbound_curveloop[1] + 1])
    gmshOCC.addSurfaceFilling(curves2)
    curves3 = gmshOCC.addCurveLoop([extbound_curveloop[0] + 3, extbound_curveloop[0] + 4, 
                                    extbound_curveloop[1] + 5, extbound_curveloop[0] + 12, 
                                    extbound_curveloop[0] + 11, extbound_curveloop[1] + 3])
    gmshOCC.addSurfaceFilling(curves3)
    curves4 = gmshOCC.addCurveLoop([extbound_curveloop[0] + 5, extbound_curveloop[0] + 6, 
                                    extbound_curveloop[1] + 7, extbound_curveloop[0] + 14, 
                                    extbound_curveloop[0] + 13, extbound_curveloop[1] + 5])
    gmshOCC.addSurfaceFilling(curves4)
    #In all of our matrix-fiber bodies, because of the linear pattern of this method's
    #meshing scheme, the matrix-fiber bodies will have the same corresponding curves
    #with tags that vary linearly, so it will be useful to have a reference curve
    #tag for each matrix ID that can be used in defining curve loops. The reference tag
    #can be chosen to have any arbitrary tag number, so define it accordingly:
    ref_tag = extbound_curveloop[0] #For mat_ID = 0, ref_tag = 49
    #Connecting butterfly prism and ellipse:
    curveLoopGenerator([ref_tag + 24, ref_tag - 8, ref_tag + 32, ref_tag + 16], 
                            [False, False, False, False], 8, [False, False, False, False])
    #Connecting ellipse and matrix:
    curveLoopGenerator([ref_tag + 40, ref_tag - 32, ref_tag + 48, ref_tag - 8], 
                            [False, False, False, False], 8, [False, False, False, False])
    #Ellipse connecting faces:
    curveLoopGenerator([ref_tag - 24, ref_tag - 8, ref_tag - 16, ref_tag - 7], 
                            [False, False, False, True], 8, [False, False, False, False])
    
    #For practicality purposes, it is also useful to define the combined surface
    #of two faces, and this will be helpful for the intermediate elliptical surfaces
    #later on. For these cases, it will be complicated to use the previously stated
    #function, so we manually define them:
    curves5 = gmshOCC.addCurveLoop([ref_tag - 24, ref_tag + 25, ref_tag, 
                                    ref_tag + 7, ref_tag + 31, ref_tag - 17])
    gmshOCC.addSurfaceFilling(curves5)
    curves6 = gmshOCC.addCurveLoop([ref_tag - 22, ref_tag + 27, ref_tag + 2, 
                                    ref_tag + 1, ref_tag + 25, ref_tag - 23])
    gmshOCC.addSurfaceFilling(curves6)
    curves7 = gmshOCC.addCurveLoop([ref_tag - 20, ref_tag + 29, ref_tag + 4, 
                                    ref_tag + 3, ref_tag + 27, ref_tag - 21])
    gmshOCC.addSurfaceFilling(curves7)
    curves8 = gmshOCC.addCurveLoop([ref_tag - 18, ref_tag + 31, ref_tag + 6, 
                                    ref_tag + 5, ref_tag + 29, ref_tag - 19])
    gmshOCC.addSurfaceFilling(curves8)
    curves9 = gmshOCC.addCurveLoop([ref_tag - 16, ref_tag + 33, ref_tag + 8, 
                                    ref_tag + 15, ref_tag + 39, ref_tag - 9])
    gmshOCC.addSurfaceFilling(curves9)
    curves10 = gmshOCC.addCurveLoop([ref_tag - 14, ref_tag + 35, ref_tag + 10, 
                                     ref_tag + 9, ref_tag + 33, ref_tag - 15])
    gmshOCC.addSurfaceFilling(curves10)
    curves11 = gmshOCC.addCurveLoop([ref_tag - 12, ref_tag + 37, ref_tag + 12, 
                                     ref_tag + 11, ref_tag + 35, ref_tag - 13])
    gmshOCC.addSurfaceFilling(curves11)
    curves12 = gmshOCC.addCurveLoop([ref_tag - 10, ref_tag + 39, ref_tag + 14, 
                                     ref_tag + 13, ref_tag + 37, ref_tag - 11])
    gmshOCC.addSurfaceFilling(curves12)
    curves13 = gmshOCC.addCurveLoop([ref_tag - 24, ref_tag - 7, ref_tag - 16, 
                                     ref_tag - 9, ref_tag - 1, ref_tag - 17])
    gmshOCC.addSurfaceFilling(curves13)
    curves14 = gmshOCC.addCurveLoop([ref_tag - 22, ref_tag - 5, ref_tag - 14, 
                                     ref_tag - 15, ref_tag - 7, ref_tag - 23])
    gmshOCC.addSurfaceFilling(curves14)
    curves15 = gmshOCC.addCurveLoop([ref_tag - 20, ref_tag - 3, ref_tag - 12, 
                                     ref_tag - 13, ref_tag - 5, ref_tag - 21])
    gmshOCC.addSurfaceFilling(curves15)
    curves16 = gmshOCC.addCurveLoop([ref_tag - 18, ref_tag - 1, ref_tag - 10, 
                                     ref_tag - 11, ref_tag - 3, ref_tag - 19])
    gmshOCC.addSurfaceFilling(curves16)
    
    #Ellipse source faces:
    curveLoopGenerator([ref_tag - 24, ref_tag + 25, ref_tag, ref_tag + 24], 
                            [False, True, False, False], 8, [False, False, False, False])
    #Ellipse target faces:
    curveLoopGenerator([ref_tag - 16, ref_tag + 33, ref_tag + 8, ref_tag + 32], 
                            [False, True, False, False], 8, [False, False, False, False])
    #Matrix source faces:
    curveLoopGenerator([ref_tag - 48, ref_tag + 41, ref_tag - 24, ref_tag + 40], 
                            [False, True, False, False], 8, [False, False, False, False])
    #Matrix target faces:
    curveLoopGenerator([ref_tag - 40, ref_tag + 49, ref_tag - 16, ref_tag + 48], 
                            [False, True, False, False], 8, [False, False, False, False])
    #Butterfly prism source faces:
    curveLoopGenerator([ref_tag, ref_tag + 1, ref_tag + 2, ref_tag + 3, 
                        ref_tag + 4, ref_tag + 5, ref_tag + 6, ref_tag + 7], 
                            [False, False, False, False, False, False, False, False], 1, 
                            [False, False, False, False, False, False, False, False])
    #Butterfly prism target faces:
    surf_references = curveLoopGenerator([ref_tag + 8, ref_tag + 9, ref_tag + 10, ref_tag + 11, 
                        ref_tag + 12, ref_tag + 13, ref_tag + 14, ref_tag + 15], 
                            [False, False, False, False, False, False, False, False], 1, 
                            [False, False, False, False, False, False, False, False])
    #Defining relevant surface loop variables (will become apparent later on):
    surf_ref_tag = surf_references[0]  #For mat_ID = 0, surf_ref_tag = 90
    
    #Nice! We've now additionally defined all of our surfaces. Next up is surface
    #loops and volume definitions. To do so, we will primarily use the utility 
    #function 'surfaceLoopGenerator,' which fully defines the volumes we would 
    #like to make. Similarly, we can use the surface reference tag, 'surf_ref_tag'
    #that can be used accordingly:
    #Outer matrix volume:
    surfaceLoopGenerator([surf_ref_tag - 89, surf_ref_tag - 60, surf_ref_tag - 53, 
                          surf_ref_tag - 61, surf_ref_tag - 17, surf_ref_tag - 9], 
                         [False, True, False, False, False, False], 8, 
                         [False, False, False, False, False, False])
    #Outer elliptical volume:
    surfaceLoopGenerator([surf_ref_tag - 53, surf_ref_tag - 68, surf_ref_tag - 81, 
                          surf_ref_tag - 69, surf_ref_tag - 33, surf_ref_tag - 25], 
                         [False, True, False, False, False, False], 8, 
                         [False, False, False, False, False, False])
    
    #Butterfly prism volume:
    vol_references = surfaceLoopGenerator([surf_ref_tag - 73, surf_ref_tag - 72, surf_ref_tag - 71, 
                          surf_ref_tag - 70, surf_ref_tag - 1, surf_ref_tag], 
                         [False, False, False, False, False, False], 1, 
                         [False, False, False, False, False, False])
    #Defining a relevant volume variable (will become apparent later on):
    vol_ref_tag = vol_references[0]  #For mat_ID = 0, vol_ref_tag = 17
    gmshOCC.synchronize()
     
    #Transfinite Settings:
    #Useful tagging variables:
    maxLTag = gmshOCC.getMaxTag(1)
    maxSTag = gmshOCC.getMaxTag(2)
    maxVTag = gmshOCC.getMaxTag(3)
    #We want to be able to specify the number of nodes on certain types of curves.
    #In this model, there are two main types of curves: source/target face curves
    #and connecting, intermediate curves. To do obtain this curve data, we concatenate 
    #a list of all the curve dimensions that are in between the sourve and target faces:
    all_curves = gmshOCC.getEntities(1)
    #All of connecting curves were generated in the surface sewing process,
    #most of whom, were out of order, so we specify these curves manually in 'sewed_curves':
    sewed_curves = [ref_tag + 165, ref_tag + 170, ref_tag + 173, 
                    ref_tag + 178, ref_tag + 183, ref_tag + 188, 
                    ref_tag + 195, ref_tag + 200, ref_tag + 205, 
                    ref_tag + 160, ref_tag + 153, ref_tag + 148, 
                    ref_tag + 143, ref_tag + 138, ref_tag + 135, 
                    ref_tag + 130, ref_tag + 80, ref_tag + 84, 
                    ref_tag + 88, ref_tag + 92]
    if mat_ID == 22:
        sewed_curves = [ref_tag + 165, ref_tag + 170, ref_tag + 173, 
                ref_tag + 178, ref_tag + 183, ref_tag + 188, 
                ref_tag + 195, ref_tag + 200, ref_tag + 205, 
                ref_tag + 160, ref_tag + 153, ref_tag + 148, 
                ref_tag + 143, ref_tag + 140, ref_tag + 135, 
                ref_tag + 130, ref_tag + 80, ref_tag + 84, 
                ref_tag + 88, ref_tag + 92]
    connecting_curves = []
    for elem in all_curves:
        boundaries = gmshMOD.getBoundary([elem])
        if boundaries[1][1] - boundaries[0][1] == 8 or boundaries[1][1] - boundaries[0][1] == 16:
            connecting_curves.append(elem)
        elif elem[1] in sewed_curves:
            connecting_curves.append(elem)
    connecting_curves_tags = [elem[1] for elem in connecting_curves]
    #Based off of whether the curve is 'connecting' or 'source' and 'target',
    #we will define a transfinite curve with numNodes determined by this functions
    #'transfinite_curves' variable:
    s_and_t = transfinite_curves[0]
    inter = transfinite_curves[1]
    for i in range(ref_tag - 48, maxLTag + 1):
        if i in connecting_curves_tags:
            gmshMESH.setTransfiniteCurve(i, inter)
        else:
            gmshMESH.setTransfiniteCurve(i, s_and_t)
    #In defining transfinite surfaces, there are also special cases (mostly related
    #to the not well-defined ellipse regions) that we have to define the outer
    #pointTags for. Additionally, we will also use the corresponding point tag
    #reference variable, 'point_ref_tag' to describe the relevant point tags for
    #each matrix ID:
    for i in range(surf_ref_tag - 89, maxSTag + 1):
         if i == surf_ref_tag - 37:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 16, 
                                                        point_ref_tag + 3, 
                                                        point_ref_tag + 20, 
                                                        point_ref_tag + 32])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref_tag - 36:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 4, 
                                                        point_ref_tag + 7, 
                                                        point_ref_tag + 20, 
                                                        point_ref_tag + 24])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref_tag - 35:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 8, 
                                                        point_ref_tag + 11, 
                                                        point_ref_tag + 24, 
                                                        point_ref_tag + 28])
             gmshMESH.setRecombine(2, i)    
         elif i == surf_ref_tag - 34:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 12, 
                                                        point_ref_tag + 15, 
                                                        point_ref_tag + 28, 
                                                        point_ref_tag + 32])
             gmshMESH.setRecombine(2, i)             
         elif i == surf_ref_tag - 45:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 3, 
                                                        point_ref_tag + 16, 
                                                        point_ref_tag + 35, 
                                                        point_ref_tag + 41])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref_tag - 44:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 4, 
                                                        point_ref_tag + 7, 
                                                        point_ref_tag + 35, 
                                                        point_ref_tag + 37])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref_tag - 43:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 8, 
                                                        point_ref_tag + 11, 
                                                        point_ref_tag + 37, 
                                                        point_ref_tag + 39])
             gmshMESH.setRecombine(2, i)    
         elif i == surf_ref_tag - 42:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 12, 
                                                        point_ref_tag + 15, 
                                                        point_ref_tag + 39, 
                                                        point_ref_tag + 41])
             gmshMESH.setRecombine(2, i)        
         elif i == surf_ref_tag - 41:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 19, 
                                                        point_ref_tag + 32, 
                                                        point_ref_tag + 43, 
                                                        point_ref_tag + 49])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref_tag - 40:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 20, 
                                                        point_ref_tag + 23, 
                                                        point_ref_tag + 43, 
                                                        point_ref_tag + 45])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref_tag - 39:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 24, 
                                                        point_ref_tag + 27, 
                                                        point_ref_tag + 45, 
                                                        point_ref_tag + 47])
             gmshMESH.setRecombine(2, i)    
         elif i == surf_ref_tag - 38:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 28, 
                                                        point_ref_tag + 31, 
                                                        point_ref_tag + 47, 
                                                        point_ref_tag + 49])
             gmshMESH.setRecombine(2, i)   
         elif i == surf_ref_tag - 1:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 35, 
                                                        point_ref_tag + 37, 
                                                        point_ref_tag + 39, 
                                                        point_ref_tag + 41])
             gmshMESH.setRecombine(2, i)    
         elif i == surf_ref_tag:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 43, 
                                                        point_ref_tag + 45, 
                                                        point_ref_tag + 47, 
                                                        point_ref_tag + 49])
             gmshMESH.setRecombine(2, i)         
         elif i == surf_ref_tag - 73:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 35, 
                                                        point_ref_tag + 41, 
                                                        point_ref_tag + 43, 
                                                        point_ref_tag + 49])
             gmshMESH.setRecombine(2, i)    
         elif i == surf_ref_tag - 72:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 35, 
                                                        point_ref_tag + 37, 
                                                        point_ref_tag + 43, 
                                                        point_ref_tag + 45])
             gmshMESH.setRecombine(2, i)   
         elif i == surf_ref_tag - 71:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 37, 
                                                        point_ref_tag + 39, 
                                                        point_ref_tag + 45, 
                                                        point_ref_tag + 47])
             gmshMESH.setRecombine(2, i)    
         elif i == surf_ref_tag - 70:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 39, 
                                                        point_ref_tag + 41, 
                                                        point_ref_tag + 47, 
                                                        point_ref_tag + 49])
             gmshMESH.setRecombine(2, i)
         else:
             gmshMESH.setTransfiniteSurface(i)
             gmshMESH.setRecombine(2, i) 
    #There is only one special case in defining transfinite volumes for the central
    #butterfly region, which we define accordingly:
    for i in range(vol_ref_tag - 16, maxVTag + 1):
          if i == maxVTag:
              gmshMESH.setTransfiniteVolume(i, [point_ref_tag + 35, 
                                                point_ref_tag + 37, 
                                                point_ref_tag + 39, 
                                                point_ref_tag + 41, 
                                                point_ref_tag + 43, 
                                                point_ref_tag + 45, 
                                                point_ref_tag + 47, 
                                                point_ref_tag + 49])
          else:
              gmshMESH.setTransfiniteVolume(i)
    
    #We're done, nice! If you read this far, go outside and eat some grass -_-
    #don't actually eat grass
    all_entities = gmshOCC.getEntities()
    gmsh.model.occ.synchronize()
    
    return all_entities

def matrixfibergeo_left(matrix_ID, transfinite_curves, angle = 30, but1 = 1/3, but2 = 2/9):
    """
     matrixfibergeo_left(matrix_ID, transfinite_curves, angle = 30, but1 = 1/3, but2 = 2/9)

    Defines geometry for one matrix-fiber set *on the left hand side of the entire layer* 
    that is based off of the SolidWorks Part files, starting with the matrix-fiber 
    set with fibers angled cross-ply at some angle 'angle' with matrix ID, 'matrix_ID', 
    such that a matrix ID of 0 corresponds to the right-most, corner-most matrix-fiber 
    body, and 'but1' and 'but2' represent the butterfly ratios, which is the ratio 
    determining how large the butterfly prism will be with respect to the elliptical 
    boundary, for the source and target butterfly prisms. 'transfinite_curves' is 
    a list of dim = 2 that contains numNodes data, such that the first element 
    represents the number of nodes on the source and target face curves, while 
    the second element represents the number of nodes on the curves in between 
    the source and target faces.

    Returns matrix-fiber set geometry entities *on the left hand side*
    """
    #Main Dimensions of one Matrix-Fiber Layer:
    height_total = 11
    length = 6
    cent = 0.22  #fiber centerline distance
    d = 0.12  #fiber diameter
    r = d/2  #fiber radius
    t = 0.2  #matrix thickness
    ang = angle  #fiber orientation in deg

    #The width of one of these matrix bodies projected onto the most
    #bottom-most edge in the geometry in mm:
    width = cent/cos(ang*pi/180)

    #The height of one of these matrix bodies projected onto the most
    #right-most edge in the geometry in mm:
    height = cent/sin(ang*pi/180)
    
    #Number of fibers to the right of this corner offset matrix:
    N_fib = 22

    #Corner offset, which is a product of the unsymmetrical fiber geometries:
    off = 0.08

    #The corner fibers in this model are always going to be weird and are edge cases
    #when defining a mesh for this geometry. Initially, it will be easier to ignore
    #these cases and start meshing the full matrix-fiber bodies themselves. Based off
    #of the SolidWorks parts, the origin (0, 0, 0) is defined as the bottom right hand
    #corner with the non-matrix-fiber body and edge with the 0.08 mm offset.

    #In dealing with this non-matrix-fiber body offset, the closest matrix body, which
    #is to the left of the defined origin (negative x-value), has some starting x
    #value for its right-most edge. After doing some trigonometry and geometry
    #breakdowns, this offset is defined as:
    start = length - width/2 - off - N_fib*width

    #Additionally, for geometry/mesh preparation, it will useful to identify these
    #matrix-fibers bodies by unique IDs. Starting off with the first offset matrix:
    mat_ID = matrix_ID
    
    #To simplify the function-calling process, gmshOCC and gmshMOD is defined:
    gmshOCC = gmsh.model.occ
    gmshMOD = gmsh.model

    #Geometry Generation
    #The new origin:
    #origin = gmshOCC.addPoint(-length, 0, -height)
    
    #Next, we wanna create the outer most boundaries of corner-most fiber. To do so,
    #we first define the source and target faces by starting at on of their midpoints.
    #This procedure will make more sense later on, when we define ellipse points.
    #Some key variables include:
    d1 = height_total - start*tan((90 - ang)*pi/180) - 24*height
    d2 = d1*sin(ang*pi/180)
    d3 = cent - d2
    off_left = d3/cos(ang*pi/180)
    off_left_corner = off_left - width/2
    start_left = length - off_left - N_fib*width
    x_start_left = -length + start_left + mat_ID*width
    y_start_left = t/2
    z_start_left = (length + x_start_left)/tan((ang)*pi/180) - height_total
    #Next, we wull define the curve boundaries. To do so, we will primarily use the utility
    #function 'matrixPrismGenerator,' which fully defines the curves we wouls like
    #to make:
    if mat_ID == 22:
        matrix = matrixPrismGenertor([[0, y_start_left, -height_total], 
                                  [-length, y_start_left, z_start_left + height]], 
                                  [off_left_corner, width/2], [t/2, t/2], [height/2, height/2])
    else:
        matrix = matrixPrismGenertor([[x_start_left + width, y_start_left, -height_total], 
                                  [-length, y_start_left, z_start_left + height]], 
                                  [width/2, width/2], [t/2, t/2], [height/2, height/2])
    #Defining relevant curve loop variables (will become apparent later on):
    matbound_curveloop = [matrix[0][1][1], matrix[16]]
    #Geometries with multiple matrix bodies will have interseting points and lines
    #at the points/surface of contact, but because overlapping entities don't
    #impact the mesh, the overlapping entities will stay in the model for
    #future model simplifications.
    
    #Next, we wanna create the geometry for the fiber volumes, which are really
    #elliptical cutouts in the matrix geometry. We start off be defining the
    #ellipse centers on each of the matrix source and target faces:
    x_ell_start = x_start_left + width/2
    y_ell_start = t/2
    z_ell_start = z_start_left + height/2
    maj_axis_source = r*sec(ang*pi/180)  #ellipse dimensions
    maj_axis_target = r*sec((90 - ang)*pi/180)
    min_axis = r
    center_source = [x_ell_start, y_ell_start, -height_total]
    center_target = [-length, y_ell_start, z_ell_start]
    #Additionally, we will defining a relevant point variable (will become apparent later on):
    point_ref_tag = gmshOCC.addPoint(center_source[0], center_source[1], center_source[2])
    #For mat_ID = 0, point_ref_tag = 17
    gmshOCC.addPoint(center_target[0], center_target[1], center_target[2])
    #Next, we define source_ell and target_ell that stores the generated ellipse
    #curve data to be able to generate coneecting lines between source and
    #target face ellipses
    source_ell = []
    target_ell = []
    for i in range(8):
        source_ell.append(gmshOCC.addEllipse(center_source[0], center_source[1], center_source[2], maj_axis_source, 
                                             min_axis, -1, i*pi/4, (i + 1)*pi/4))
        #print("Ellipse arcs on source face: ", source_ell, "Ellipse arcs on target face: ", target_ell)
    for i in range(8):
        target_ell.append(gmshOCC.addEllipse(center_target[0], center_target[1], center_target[2], maj_axis_target, 
                                             min_axis, -1, i*pi/4, (i + 1)*pi/4))
        #print("Ellipse arcs on source face: ", source_ell, "Ellipse arcs on target face: ", target_ell)
        #Because the addEllipse function does not allow you to define the orientation
        #in which the ellipse is defined, we must automatically rotate its curves
        #to define it for our model:
        gmshOCC.rotate([(1, target_ell[-1])], center_target[0], center_target[1], center_target[2], 0, -1, 0, pi/2)
    #Current gmsh model needs to be synchronized onto OpenCASCADE CAD representation
    #in order to use stored source_ell and target_ell data
    #NOTE, generally you want to be able to reduce the number of times you
    #synchronize your model because it generally uses a nontrivial amount of memory
    gmshOCC.synchronize()
    
    s_ellipse_boundary = []
    t_ellipse_boundary = []
    for source, target in zip(source_ell, target_ell):
        s_ellipse_boundary.append(gmshMOD.getBoundary([(1, source)]))
        t_ellipse_boundary.append(gmshMOD.getBoundary([(1, target)]))
        ellipse_source_boundary = gmshMOD.getBoundary([(1, source)])
        ellipse_target_boundary = gmshMOD.getBoundary([(1, target)])
        #Connecting the source and target base faces with lines:
        gmshOCC.addLine(ellipse_source_boundary[0][1], ellipse_target_boundary[0][1])
    
    #Next, we need to create our third, final primary structural component of our
    #mesh: the Butterfly Prism in the center of the ellipse structure. To do so,
    #we define some key geometric variables:
    butterfly_ratio_1 = but1
    butterfly_ratio_2 = but2
    square_source =  butterfly_ratio_1*maj_axis_source
    square_target =  butterfly_ratio_2*maj_axis_target
    #Next, we wull define the curve boundaries. To do so, we will primarily use the utility
    #function 'matrixPrismGenerator,' which fully defines the curves we wouls like
    #to make:
    extruded_entities = matrixPrismGenertor([[center_source[0] + square_source, center_source[1], 
                       center_source[2]], [center_target[0], center_target[1], 
                                           center_target[2] + square_target]], 
                                           [square_source, square_source], 
                                           [square_source, square_target], 
                                           [square_target, square_target])
    #Defining relevant curve loop variables (will become apparent later on):
    extbound_curveloop = [extruded_entities[0][1][1], extruded_entities[16]]

    gmshOCC.synchronize()
    #Now that we've created all of the main meshing bodies, we have to define the
    #lines that connect these bodies. To do, we have to do a lot of data sorting
    #with the curve entities to obtain the boundary points that define these curves.
    #First, we get specifically the dim = 1 extruded entitiy data from the previous
    #command defined in 'extruded_entities':
    extruded_curves = [dimTagReturner(1, elem) if type(elem) is list else elem for elem in extruded_entities]
    extruded_dimtags = []
    for elem in extruded_curves:
        if type(elem) is list:
            extruded_dimtags.append(elem[0])
        else:
            extruded_dimtags.append(elem)
    extruded_dimtags_source = extruded_dimtags[:8]
    extruded_dimtags_target = extruded_dimtags[8:len(extruded_dimtags)]
     
    #Synchronize model to access .model function 'getBoundary'
    gmshOCC.synchronize()
    
    #Using 'getBoundary' to obtain boundary points of curves defined in
    #'extruded_dimtags_source' and 'extruded_dimtags_target'
    s_extruded_boundary = []
    t_extruded_boundary = []
    for source, target in zip(extruded_dimtags_source, extruded_dimtags_target):
        s_extruded_boundary.append(gmshMOD.getBoundary([source]))
        t_extruded_boundary.append(gmshMOD.getBoundary([target]))
    #Putting everything together to draw out the boundary point tags to use to 
    #add connecting lines between butterfly prism and ellipse boundaries
    s_extruded_boundary[-1].reverse()
    for ellipse, extruded in zip(s_ellipse_boundary, s_extruded_boundary):
        ellipse_source_point = ellipse[0][1]
        extruded_source_point = extruded[0][1]
        gmshOCC.addLine(ellipse_source_point, extruded_source_point)
    t_extruded_boundary[-1].reverse()
    for ellipse, extruded in zip(t_ellipse_boundary, t_extruded_boundary):
        ellipse_target_point = ellipse[0][1]
        extruded_target_point = extruded[0][1]
        gmshOCC.addLine(ellipse_target_point, extruded_target_point)
    
    #Next, we get the corresponding curve entitity data from the main matrix body
    #and we do almost the same exact procedure:
    matrix_curves = [dimTagReturner(1, elem) if type(elem) is list else elem for elem in matrix]
    matrix_dimtags = []
    for elem in matrix_curves:
        if type(elem) is list:
            matrix_dimtags.append(elem[0])
        else:
            matrix_dimtags.append(elem)
    matrix_dimtags_source = matrix_dimtags[:8]
    matrix_dimtags_target = matrix_dimtags[8:len(extruded_dimtags)]
    
    gmshOCC.synchronize()
    
    #Using 'getBoundary' to obtain boundary points of curves defined in
    #'extruded_dimtags_source' and 'extruded_dimtags_target'
    s_matrix_boundary = []
    t_matrix_boundary = []
    for source, target in zip(matrix_dimtags_source, matrix_dimtags_target):
        s_matrix_boundary.append(gmshMOD.getBoundary([source]))
        t_matrix_boundary.append(gmshMOD.getBoundary([target]))
    #Putting everything together to draw out the boundary point tags to use to 
    #add connecting lines between butterfly prism and ellipse boundaries
    s_matrix_boundary[-1].reverse()
    for ellipse, matrix in zip(s_ellipse_boundary, s_matrix_boundary):
        ellipse_source_point = ellipse[0][1]
        matrix_source_point = matrix[0][1]
        gmshOCC.addLine(ellipse_source_point, matrix_source_point)
    t_matrix_boundary[-1].reverse()
    for ellipse, matrix in zip(t_ellipse_boundary, t_matrix_boundary):
        ellipse_target_point = ellipse[0][1]
        matrix_target_point = matrix[0][1]
        gmshOCC.addLine(ellipse_target_point, matrix_target_point)
    
    #Great, now we essentially have all of our points and curves fully defined!
    #Now, we actually have to use these points/curves to define curveloops, surfaces,
    #and volumes. Let's focus primarily on the outer matrix body first. To do so, 
    #we will primarily use the utility function 'curveLoopGenerator,' which fully
    #defines the surfaces we would like to make:
    #Outer matrix boundary:
    curveLoopGenerator([matbound_curveloop[0], matbound_curveloop[1], 
                        matbound_curveloop[0] + 8, matbound_curveloop[1] + 1], 
                           [False, False, False, True], 8, [False, False, False, False])
    #The butterfly prism, ellipses, and intermediate surfaces require a similar 
    #procedure as well:
    #Butterfly prism (8 individual surfaces):
    curveLoopGenerator([extbound_curveloop[0], extbound_curveloop[1], 
                        extbound_curveloop[0] + 8, extbound_curveloop[1] + 1], 
                            [False, False, False, True], 8, [False, False, False, False])
    #Butterfly prism (4 grouped surfaces):
    curves1 = gmshOCC.addCurveLoop([extbound_curveloop[0] + 7, extbound_curveloop[0], 
                                    extbound_curveloop[1] + 1, extbound_curveloop[0] + 8, 
                                    extbound_curveloop[0] + 15, extbound_curveloop[1] + 7])
    gmshOCC.addSurfaceFilling(curves1)
    curves2 = gmshOCC.addCurveLoop([extbound_curveloop[0] + 1, extbound_curveloop[0] + 2, 
                                    extbound_curveloop[1] + 3, extbound_curveloop[0] + 10, 
                                    extbound_curveloop[0] + 9, extbound_curveloop[1] + 1])
    gmshOCC.addSurfaceFilling(curves2)
    curves3 = gmshOCC.addCurveLoop([extbound_curveloop[0] + 3, extbound_curveloop[0] + 4, 
                                    extbound_curveloop[1] + 5, extbound_curveloop[0] + 12, 
                                    extbound_curveloop[0] + 11, extbound_curveloop[1] + 3])
    gmshOCC.addSurfaceFilling(curves3)
    curves4 = gmshOCC.addCurveLoop([extbound_curveloop[0] + 5, extbound_curveloop[0] + 6, 
                                    extbound_curveloop[1] + 7, extbound_curveloop[0] + 14, 
                                    extbound_curveloop[0] + 13, extbound_curveloop[1] + 5])
    gmshOCC.addSurfaceFilling(curves4)
    #In all of our matrix-fiber bodies, because of the linear pattern of this method's
    #meshing scheme, the matrix-fiber bodies will have the same corresponding curves
    #with tags that vary linearly, so it will be useful to have a reference curve
    #tag for each matrix ID that can be used in defining curve loops. The reference tag
    #can be chosen to have any arbitrary tag number, so define it accordingly:
    ref_tag = extbound_curveloop[0] #For mat_ID = 0, ref_tag = 49
    #Connecting butterfly prism and ellipse:
    curveLoopGenerator([ref_tag + 24, ref_tag - 8, ref_tag + 32, ref_tag + 16], 
                            [False, False, False, False], 8, [False, False, False, False])
    #Connecting ellipse and matrix:
    curveLoopGenerator([ref_tag + 40, ref_tag - 32, ref_tag + 48, ref_tag - 8], 
                            [False, False, False, False], 8, [False, False, False, False])
    #Ellipse connecting faces:
    curveLoopGenerator([ref_tag - 24, ref_tag - 8, ref_tag - 16, ref_tag - 7], 
                            [False, False, False, True], 8, [False, False, False, False])
    
    #For practicality purposes, it is also useful to define the combined surface
    #of two faces, and this will be helpful for the intermediate elliptical surfaces
    #later on. For these cases, it will be complicated to use the previously stated
    #function, so we manually define them:
    curves5 = gmshOCC.addCurveLoop([ref_tag - 24, ref_tag + 25, ref_tag, 
                                    ref_tag + 7, ref_tag + 31, ref_tag - 17])
    gmshOCC.addSurfaceFilling(curves5)
    curves6 = gmshOCC.addCurveLoop([ref_tag - 22, ref_tag + 27, ref_tag + 2, 
                                    ref_tag + 1, ref_tag + 25, ref_tag - 23])
    gmshOCC.addSurfaceFilling(curves6)
    curves7 = gmshOCC.addCurveLoop([ref_tag - 20, ref_tag + 29, ref_tag + 4, 
                                    ref_tag + 3, ref_tag + 27, ref_tag - 21])
    gmshOCC.addSurfaceFilling(curves7)
    curves8 = gmshOCC.addCurveLoop([ref_tag - 18, ref_tag + 31, ref_tag + 6, 
                                    ref_tag + 5, ref_tag + 29, ref_tag - 19])
    gmshOCC.addSurfaceFilling(curves8)
    curves9 = gmshOCC.addCurveLoop([ref_tag - 16, ref_tag + 33, ref_tag + 8, 
                                    ref_tag + 15, ref_tag + 39, ref_tag - 9])
    gmshOCC.addSurfaceFilling(curves9)
    curves10 = gmshOCC.addCurveLoop([ref_tag - 14, ref_tag + 35, ref_tag + 10, 
                                     ref_tag + 9, ref_tag + 33, ref_tag - 15])
    gmshOCC.addSurfaceFilling(curves10)
    curves11 = gmshOCC.addCurveLoop([ref_tag - 12, ref_tag + 37, ref_tag + 12, 
                                     ref_tag + 11, ref_tag + 35, ref_tag - 13])
    gmshOCC.addSurfaceFilling(curves11)
    curves12 = gmshOCC.addCurveLoop([ref_tag - 10, ref_tag + 39, ref_tag + 14, 
                                     ref_tag + 13, ref_tag + 37, ref_tag - 11])
    gmshOCC.addSurfaceFilling(curves12)
    curves13 = gmshOCC.addCurveLoop([ref_tag - 24, ref_tag - 7, ref_tag - 16, 
                                     ref_tag - 9, ref_tag - 1, ref_tag - 17])
    gmshOCC.addSurfaceFilling(curves13)
    curves14 = gmshOCC.addCurveLoop([ref_tag - 22, ref_tag - 5, ref_tag - 14, 
                                     ref_tag - 15, ref_tag - 7, ref_tag - 23])
    gmshOCC.addSurfaceFilling(curves14)
    curves15 = gmshOCC.addCurveLoop([ref_tag - 20, ref_tag - 3, ref_tag - 12, 
                                     ref_tag - 13, ref_tag - 5, ref_tag - 21])
    gmshOCC.addSurfaceFilling(curves15)
    curves16 = gmshOCC.addCurveLoop([ref_tag - 18, ref_tag - 1, ref_tag - 10, 
                                     ref_tag - 11, ref_tag - 3, ref_tag - 19])
    gmshOCC.addSurfaceFilling(curves16)
    
    #Ellipse source faces:
    curveLoopGenerator([ref_tag - 24, ref_tag + 25, ref_tag, ref_tag + 24], 
                            [False, True, False, False], 8, [False, False, False, False])
    #Ellipse target faces:
    curveLoopGenerator([ref_tag - 16, ref_tag + 33, ref_tag + 8, ref_tag + 32], 
                            [False, True, False, False], 8, [False, False, False, False])
    #Matrix source faces:
    curveLoopGenerator([ref_tag - 48, ref_tag + 41, ref_tag - 24, ref_tag + 40], 
                            [False, True, False, False], 8, [False, False, False, False])
    #Matrix target faces:
    curveLoopGenerator([ref_tag - 40, ref_tag + 49, ref_tag - 16, ref_tag + 48], 
                            [False, True, False, False], 8, [False, False, False, False])
    #Butterfly prism source faces:
    curveLoopGenerator([ref_tag, ref_tag + 1, ref_tag + 2, ref_tag + 3, 
                        ref_tag + 4, ref_tag + 5, ref_tag + 6, ref_tag + 7], 
                            [False, False, False, False, False, False, False, False], 1, 
                            [False, False, False, False, False, False, False, False])
    #Butterfly prism target faces:
    surf_references = curveLoopGenerator([ref_tag + 8, ref_tag + 9, ref_tag + 10, ref_tag + 11, 
                        ref_tag + 12, ref_tag + 13, ref_tag + 14, ref_tag + 15], 
                            [False, False, False, False, False, False, False, False], 1, 
                            [False, False, False, False, False, False, False, False])
    #Defining relevant surface loop variables (will become apparent later on):
    surf_ref_tag = surf_references[0]  #For mat_ID = 0, surf_ref_tag = 90

    #Nice! We've now additionally defined all of our surfaces. Next up is surface
    #loops and volume definitions. To do so, we will primarily use the utility 
    #function 'surfaceLoopGenerator,' which fully defines the volumes we would 
    #like to make. Similarly, we can use the surface reference tag, 'surf_ref_tag'
    #that can be used accordingly:
    #Outer matrix volume:
    surfaceLoopGenerator([surf_ref_tag - 89, surf_ref_tag - 60, surf_ref_tag - 53, 
                          surf_ref_tag - 61, surf_ref_tag - 17, surf_ref_tag - 9], 
                         [False, True, False, False, False, False], 8, 
                         [False, False, False, False, False, False])
    #Outer elliptical volume:
    surfaceLoopGenerator([surf_ref_tag - 53, surf_ref_tag - 68, surf_ref_tag - 81, 
                          surf_ref_tag - 69, surf_ref_tag - 33, surf_ref_tag - 25], 
                         [False, True, False, False, False, False], 8, 
                         [False, False, False, False, False, False])
    
    #Butterfly prism volume:
    vol_references = surfaceLoopGenerator([surf_ref_tag - 73, surf_ref_tag - 72, surf_ref_tag - 71, 
                          surf_ref_tag - 70, surf_ref_tag - 1, surf_ref_tag], 
                         [False, False, False, False, False, False], 1, 
                         [False, False, False, False, False, False])
    #Defining a relevant volume variable (will become apparent later on):
    vol_ref_tag = vol_references[0]  #For mat_ID = 0, vol_ref_tag = 17
    gmshOCC.synchronize()

    #Transfinite Settings:
    #Useful tagging variables:
    maxLTag = gmshOCC.getMaxTag(1)
    maxSTag = gmshOCC.getMaxTag(2)
    maxVTag = gmshOCC.getMaxTag(3)
    #We want to be able to specify the number of nodes on certain types of curves.
    #In this model, there are two main types of curves: source/target face curves
    #and connecting, intermediate curves. To obtain this curve data, we concatenate 
    #a list of all the curve dimensions that are in between the sourve and target faces:
    all_curves = gmshOCC.getEntities(1)
    #All of connecting curves were generated in the surface sewing process,
    #most of whom, were out of order, so we specify these curves manually in 'sewed_curves':
    sewed_curves = [ref_tag + 80, ref_tag + 84, ref_tag + 88, ref_tag + 92,
                    ref_tag + 128, ref_tag + 133, ref_tag + 140, ref_tag + 145,
                    ref_tag + 150, ref_tag + 155, ref_tag + 158, ref_tag + 163,
                    ref_tag + 168, ref_tag + 175, ref_tag + 180, ref_tag + 185,
                    ref_tag + 190, ref_tag + 193, ref_tag + 198, ref_tag + 203]
    if mat_ID == 22:
        sewed_curves = [ref_tag + 80, ref_tag + 84, ref_tag + 88, ref_tag + 92,
                    ref_tag + 128, ref_tag + 133, ref_tag + 140, ref_tag + 145,
                    ref_tag + 150, ref_tag + 155, ref_tag + 160, ref_tag + 163,
                    ref_tag + 168, ref_tag + 175, ref_tag + 180, ref_tag + 185,
                    ref_tag + 190, ref_tag + 193, ref_tag + 198, ref_tag + 203]
    connecting_curves = []
    for elem in all_curves:
        boundaries = gmshMOD.getBoundary([elem])
        if boundaries[1][1] - boundaries[0][1] == 8 or boundaries[1][1] - boundaries[0][1] == 16:
            connecting_curves.append(elem)
        elif elem[1] in sewed_curves:
            connecting_curves.append(elem)
    connecting_curves_tags = [elem[1] for elem in connecting_curves]
    #Based off of whether the curve is 'connecting' or 'source' and 'target',
    #we will define a transfinite curve with numNodes determined by this functions
    #'transfinite_curves' variable:
    s_and_t = transfinite_curves[0]
    inter = transfinite_curves[1]
    for i in range(ref_tag - 48, maxLTag + 1):
        if i in connecting_curves_tags:
            gmshMESH.setTransfiniteCurve(i, inter)
        else:
            gmshMESH.setTransfiniteCurve(i, s_and_t)
    #In defining transfinite surfaces, there are also special cases (mostly related
    #to the not well-defined ellipse regions) that we have to define the outer
    #pointTags for. Additionally, we will also use the corresponding point tag
    #reference variable, 'point_ref_tag' to describe the relevant point tags for
    #each matrix ID:
    for i in range(surf_ref_tag - 89, maxSTag + 1):
         if i == surf_ref_tag - 37:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 16, 
                                                        point_ref_tag + 3, 
                                                        point_ref_tag + 20, 
                                                        point_ref_tag + 32])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref_tag - 36:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 4, 
                                                        point_ref_tag + 7, 
                                                        point_ref_tag + 20, 
                                                        point_ref_tag + 24])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref_tag - 35:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 8, 
                                                        point_ref_tag + 11, 
                                                        point_ref_tag + 24, 
                                                        point_ref_tag + 28])
             gmshMESH.setRecombine(2, i)    
         elif i == surf_ref_tag - 34:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 12, 
                                                        point_ref_tag + 15, 
                                                        point_ref_tag + 28, 
                                                        point_ref_tag + 32])
             gmshMESH.setRecombine(2, i)             
         elif i == surf_ref_tag - 45:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 3, 
                                                        point_ref_tag + 16, 
                                                        point_ref_tag + 35, 
                                                        point_ref_tag + 41])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref_tag - 44:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 4, 
                                                        point_ref_tag + 7, 
                                                        point_ref_tag + 35, 
                                                        point_ref_tag + 37])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref_tag - 43:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 8, 
                                                        point_ref_tag + 11, 
                                                        point_ref_tag + 37, 
                                                        point_ref_tag + 39])
             gmshMESH.setRecombine(2, i)    
         elif i == surf_ref_tag - 42:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 12, 
                                                        point_ref_tag + 15, 
                                                        point_ref_tag + 39, 
                                                        point_ref_tag + 41])
             gmshMESH.setRecombine(2, i)        
         elif i == surf_ref_tag - 41:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 19, 
                                                        point_ref_tag + 32, 
                                                        point_ref_tag + 43, 
                                                        point_ref_tag + 49])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref_tag - 40:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 20, 
                                                        point_ref_tag + 23, 
                                                        point_ref_tag + 43, 
                                                        point_ref_tag + 45])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref_tag - 39:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 24, 
                                                        point_ref_tag + 27, 
                                                        point_ref_tag + 45, 
                                                        point_ref_tag + 47])
             gmshMESH.setRecombine(2, i)    
         elif i == surf_ref_tag - 38:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 28, 
                                                        point_ref_tag + 31, 
                                                        point_ref_tag + 47, 
                                                        point_ref_tag + 49])
             gmshMESH.setRecombine(2, i)   
         elif i == surf_ref_tag - 1:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 35, 
                                                        point_ref_tag + 37, 
                                                        point_ref_tag + 39, 
                                                        point_ref_tag + 41])
             gmshMESH.setRecombine(2, i)    
         elif i == surf_ref_tag:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 43, 
                                                        point_ref_tag + 45, 
                                                        point_ref_tag + 47, 
                                                        point_ref_tag + 49])
             gmshMESH.setRecombine(2, i)         
         elif i == surf_ref_tag - 73:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 35, 
                                                        point_ref_tag + 41, 
                                                        point_ref_tag + 43, 
                                                        point_ref_tag + 49])
             gmshMESH.setRecombine(2, i)    
         elif i == surf_ref_tag - 72:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 35, 
                                                        point_ref_tag + 37, 
                                                        point_ref_tag + 43, 
                                                        point_ref_tag + 45])
             gmshMESH.setRecombine(2, i)   
         elif i == surf_ref_tag - 71:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 37, 
                                                        point_ref_tag + 39, 
                                                        point_ref_tag + 45, 
                                                        point_ref_tag + 47])
             gmshMESH.setRecombine(2, i)    
         elif i == surf_ref_tag - 70:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 39, 
                                                        point_ref_tag + 41, 
                                                        point_ref_tag + 47, 
                                                        point_ref_tag + 49])
             gmshMESH.setRecombine(2, i)

         else:
             gmshMESH.setTransfiniteSurface(i)
             gmshMESH.setRecombine(2, i) 
    #There is only one special case in defining transfinite volumes for the central
    #butterfly region, which we define accordingly:
    for i in range(vol_ref_tag - 16, maxVTag + 1):
          if i == maxVTag:
              gmshMESH.setTransfiniteVolume(i, [point_ref_tag + 35, 
                                                point_ref_tag + 37, 
                                                point_ref_tag + 39, 
                                                point_ref_tag + 41, 
                                                point_ref_tag + 43, 
                                                point_ref_tag + 45, 
                                                point_ref_tag + 47, 
                                                point_ref_tag + 49])
          else:
              gmshMESH.setTransfiniteVolume(i)

    #We're done, nice! If you read this far, go outside and eat some grass -_- AGAIN
    #don't actually eat grass
    all_entities = gmshOCC.getEntities()
    gmshOCC.synchronize()
    
    return all_entities

def matrixfibergeo_central(transfinite_curves, matrix_ID = 23, angle = 30, but1 = 2/9, but2 = 2/9):
    #Key dimensions explained/defined in previous functions:
    height_total = 11
    length = 6
    cent = 0.22  #fiber centerline distance
    d = 0.12  #fiber diameter
    r = d/2  #fiber radius
    t = 0.2  #matrix thickness
    ang = angle  #fiber orientation in deg

    #The width of one of these matrix bodies projected onto the most
    #bottom-most edge in the geometry in mm:
    width = cent/cos(ang*pi/180)

    #The height of one of these matrix bodies projected onto the most
    #right-most edge in the geometry in mm:
    height = cent/sin(ang*pi/180)
    
    #Number of fibers to the right of this corner offset matrix:
    N_fib = 22

    #Corner offset, which is a product of the unsymmetrical fiber geometries:
    off = 0.08

    #The corner fibers in this model are always going to be weird and are edge cases
    #when defining a mesh for this geometry. Initially, it will be easier to ignore
    #these cases and start meshing the full matrix-fiber bodies themselves. Based off
    #of the SolidWorks parts, the origin (0, 0, 0) is defined as the bottom right hand
    #corner with the non-matrix-fiber body and edge with the 0.08 mm offset.

    #In dealing with this non-matrix-fiber body offset, the closest matrix body, which
    #is to the left of the defined origin (negative x-value), has some starting x
    #value for its right-most edge. After doing some trigonometry and geometry
    #breakdowns, this offset is defined as:
    start = length - width/2 - off - N_fib*width

    #Additionally, for geometry/mesh preparation, it will useful to identify these
    #matrix-fibers bodies by unique IDs. Starting off with the first offset matrix:
    mat_ID = matrix_ID
    
    #To simplify the function-calling process, gmshOCC and gmshMOD is defined:
    gmshOCC = gmsh.model.occ
    gmshMOD = gmsh.model

    #Geometry Generation
    #The new origin:
    #origin = gmshOCC.addPoint(-length, 0, -height)
    
    #Next, we wanna create the outer most boundaries of corner-most fiber. To do so,
    #we first define the source and target faces by starting at on of their midpoints.
    #This procedure will make more sense later on, when we define ellipse points.
    #Some key variables include:
    x_start = -start - mat_ID*width
    y_start = t/2
    z_start = -abs(x_start)*tan((90 - ang)*pi/180)
    m = tan((90 - ang)*pi/180)
    TRC_off = height_total - abs(z_start) - height
    BLC_off = m*(-length) - z_start
    
    #Source face points:
    point1 = gmshOCC.addPoint(0, y_start, z_start)
    point_ref_tag = point1
    point2 = gmshOCC.addPoint(0, y_start + t/2, z_start)
    point3 = gmshOCC.addPoint(0, y_start + t/2, z_start - height/2)
    point4 = gmshOCC.addPoint(0, y_start + t/2, z_start - height - TRC_off)
    point5 = gmshOCC.addPoint(0, y_start, z_start - height - TRC_off)
    point6 = gmshOCC.addPoint(0, y_start - t/2, z_start - height - TRC_off)
    point7 = gmshOCC.addPoint(0, y_start - t/2, z_start - height/2)
    point8 = gmshOCC.addPoint(0, y_start - t/2, z_start)
    #Source face lines:
    source_mat_curves = []
    source_mat_curves.append(gmshOCC.addLine(point1, point2)) 
    source_mat_curves.append(gmshOCC.addLine(point2, point3)) 
    source_mat_curves.append(gmshOCC.addLine(point3, point4))
    source_mat_curves.append(gmshOCC.addLine(point4, point5))
    source_mat_curves.append(gmshOCC.addLine(point5, point6))
    source_mat_curves.append(gmshOCC.addLine(point6, point7))
    source_mat_curves.append(gmshOCC.addLine(point7, point8))
    source_mat_curves.append(gmshOCC.addLine(point8, point1))
    curve_ref = source_mat_curves[0]
    #Additional key variables:
    d1 = height_total - start*tan((90 - ang)*pi/180) - 24*height
    d2 = d1*sin(ang*pi/180)
    d3 = cent - d2
    off_left = d3/cos(ang*pi/180)
    #off_left_corner = off_left - width/2
    start_left = length - off_left - N_fib*width
    x_start_left = -length + start_left + mat_ID*width
    y_start_left = t/2
    z_start_left = (length + x_start_left)/tan((ang)*pi/180) - height_total
    #Target face points:
    point9 = gmshOCC.addPoint(-length, y_start_left, z_start_left + height + BLC_off)
    point10 = gmshOCC.addPoint(-length, y_start_left + t/2, z_start_left + height + BLC_off)
    point11 = gmshOCC.addPoint(-length, y_start_left + t/2, z_start_left + height/2)
    point12 = gmshOCC.addPoint(-length, y_start_left + t/2, z_start_left)
    point13 = gmshOCC.addPoint(-length, y_start_left, z_start_left)
    point14 = gmshOCC.addPoint(-length, y_start_left - t/2, z_start_left)
    point15 = gmshOCC.addPoint(-length, y_start_left - t/2, z_start_left + height/2)
    point16 = gmshOCC.addPoint(-length, y_start_left - t/2, z_start_left + height + BLC_off)
    #Target face lines:
    target_mat_curves = []
    target_mat_curves.append(gmshOCC.addLine(point9, point10))
    target_mat_curves.append(gmshOCC.addLine(point10, point11))
    target_mat_curves.append(gmshOCC.addLine(point11, point12))
    target_mat_curves.append(gmshOCC.addLine(point12, point13))
    target_mat_curves.append(gmshOCC.addLine(point13, point14))
    target_mat_curves.append(gmshOCC.addLine(point14, point15))
    target_mat_curves.append(gmshOCC.addLine(point15, point16))
    target_mat_curves.append(gmshOCC.addLine(point16, point9))
    #Connecting source and target faces
    gmshOCC.addLine(point1, point9)
    gmshOCC.addLine(point2, point10)
    gmshOCC.addLine(point3, point11)
    gmshOCC.addLine(point4, point12)
    gmshOCC.addLine(point5, point13)
    gmshOCC.addLine(point6, point14)
    gmshOCC.addLine(point7, point15)
    gmshOCC.addLine(point8, point16)
    
    #Next, we wanna create the geometry for the fiber volumes, which are really
    #elliptical cutouts in the matrix geometry. We start off be defining the
    #ellipse centers on each of the matrix source and target faces:
    #x_ell_start = x_start - width/2
    y_ell_start = t/2
    z_ell_start = z_start - height/2
    maj_axis = r*sec((90 - ang)*pi/180)  #ellipse dimensions
    min_axis = r
    center_source = [0, y_ell_start, z_ell_start]
    
    #Additionally, we will defining a relevant point variable (will become apparent later on):
    gmshOCC.addPoint(center_source[0], center_source[1], center_source[2])
    #For mat_ID = 0, point_ref_tag = 17
    #Next, we define source_ell and target_ell that stores the generated ellipse
    #curve data to be able to generate coneecting lines between source and
    #target face ellipses
    source_ell = []
    target_ell = []
    for i in range(8):
        source_ell.append(gmshOCC.addEllipse(center_source[0], center_source[1], center_source[2], maj_axis, 
                                             min_axis, -1, i*pi/4, (i + 1)*pi/4))
        #Because the addEllipse function does not allow you to define the orientation
        #in which the ellipse is defined, we must automatically rotate its curves
        #to define it for our model:
        gmshOCC.rotate([(1, source_ell[-1])], center_source[0], center_source[1], center_source[2], 0, -1, 0, pi/2)
    
    #x_ell_start = x_start_left + width/2
    y_ell_start = t/2
    z_ell_start = z_start_left + height/2
    center_target = [-length, y_ell_start, z_ell_start]
    gmshOCC.addPoint(center_target[0], center_target[1], center_target[2])
    for i in range(8):
        target_ell.append(gmshOCC.addEllipse(center_target[0], center_target[1], center_target[2], maj_axis, 
                                             min_axis, -1, i*pi/4, (i + 1)*pi/4))
        #Because the addEllipse function does not allow you to define the orientation
        #in which the ellipse is defined, we must automatically rotate its curves
        #to define it for our model:
        gmshOCC.rotate([(1, target_ell[-1])], center_target[0], center_target[1], center_target[2], 0, -1, 0, pi/2)
    #Current gmsh model needs to be synchronized onto OpenCASCADE CAD representation
    #in order to use stored source_ell and target_ell data
    #NOTE, generally you want to be able to reduce the number of times you
    #synchronize your model because it generally uses a nontrivial amount of memory
    gmshOCC.synchronize()
    
    s_ellipse_boundary = []
    t_ellipse_boundary = []
    for source, target in zip(source_ell, target_ell):
        s_ellipse_boundary.append(gmshMOD.getBoundary([(1, source)]))
        t_ellipse_boundary.append(gmshMOD.getBoundary([(1, target)]))
        ellipse_source_boundary = gmshMOD.getBoundary([(1, source)])
        ellipse_target_boundary = gmshMOD.getBoundary([(1, target)])
        #Connecting the source and target base faces with lines:
        gmshOCC.addLine(ellipse_source_boundary[0][1], ellipse_target_boundary[0][1])
    
    #Next, we need to create our third, final primary structural component of our
    #mesh: the Butterfly Prism in the center of the ellipse structure. To do so,
    #we define some key geometric variables:
    butterfly_ratio_1 = but1
    butterfly_ratio_2 = but2
    square_source =  butterfly_ratio_1*maj_axis
    square_target =  butterfly_ratio_2*maj_axis
    #Next, we wull define the curve boundaries. To do so, we will primarily use the utility
    #function 'matrixPrismGenerator,' which fully defines the curves we wouls like
    #to make:
    
    extrusions = []
    #Defining the source point and creating extrusions to create base for source face:
    center1_start = gmshOCC.addPoint(center_source[0], center_source[1], center_source[2] + square_source)
    extrusions.append(gmshOCC.extrude([(0, center1_start)], 0, square_source, 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, -square_source))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, -square_source))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, -square_source, 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, -square_source, 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, square_source))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, square_source))
    extrusions.append(tuple([1, gmshOCC.addLine(gmshOCC.getMaxTag(0), center1_start)]))
    #Defining the target point and creating extrusions to create base for target face:
    center2_start = gmshOCC.addPoint(center_target[0], center_target[1], center_target[2] + square_target)
    extrusions.append(gmshOCC.extrude([(0, center2_start)], 0, square_target, 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, -square_target))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, -square_target))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, -square_target, 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, -square_target, 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, square_target))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, square_target))
    extrusions.append(tuple([1, gmshOCC.addLine(gmshOCC.getMaxTag(0), center2_start)]))
    #Connecting the source and target base faces with lines:
    for i in range(8):
        extrusions.append(gmshOCC.addLine(center1_start + i, center2_start + i))
                         
    gmshOCC.synchronize()
    #Now that we've created all of the main meshing bodies, we have to define the
    #lines that connect these bodies. To do, we have to do a lot of data sorting
    #with the curve entities to obtain the boundary points that define these curves.
    #First, we get specifically the dim = 1 extruded entitiy data from the previous
    #command defined in 'extruded_entities':
    extruded_curves = [dimTagReturner(1, elem) if type(elem) is list else elem for elem in extrusions]
    extruded_dimtags = []
    for elem in extruded_curves:
        if type(elem) is list:
            extruded_dimtags.append(elem[0])
        else:
            extruded_dimtags.append(elem)
    extruded_dimtags_source = extruded_dimtags[:8]
    extruded_dimtags_target = extruded_dimtags[8:len(extruded_dimtags)]
             
    #Synchronize model to access .model function 'getBoundary'
    gmshOCC.synchronize()
    
    #Using 'getBoundary' to obtain boundary points of curves defined in
    #'extruded_dimtags_source' and 'extruded_dimtags_target'
    s_extruded_boundary = []
    t_extruded_boundary = []
    for source, target in zip(extruded_dimtags_source, extruded_dimtags_target):
        s_extruded_boundary.append(gmshMOD.getBoundary([source]))
        t_extruded_boundary.append(gmshMOD.getBoundary([target]))
    #Putting everything together to draw out the boundary point tags to use to 
    #add connecting lines between butterfly prism and ellipse boundaries
    s_extruded_boundary[-1].reverse()
    for ellipse, extruded in zip(s_ellipse_boundary, s_extruded_boundary):
        ellipse_source_point = ellipse[0][1]
        extruded_source_point = extruded[0][1]
        gmshOCC.addLine(ellipse_source_point, extruded_source_point)
    t_extruded_boundary[-1].reverse()
    for ellipse, extruded in zip(t_ellipse_boundary, t_extruded_boundary):
        ellipse_target_point = ellipse[0][1]
        extruded_target_point = extruded[0][1]
        gmshOCC.addLine(ellipse_target_point, extruded_target_point)
        
    gmshOCC.synchronize()
    
    #Using 'getBoundary' to obtain boundary points of curves defined in
    #'extruded_dimtags_source' and 'extruded_dimtags_target'
    s_matrix_boundary = []
    t_matrix_boundary = []
    for source, target in zip(source_mat_curves, target_mat_curves):
        s_matrix_boundary.append(gmshMOD.getBoundary([(1, source)]))
        t_matrix_boundary.append(gmshMOD.getBoundary([(1, target)]))
    #Putting everything together to draw out the boundary point tags to use to 
    #add connecting lines between butterfly prism and ellipse boundaries
    s_matrix_boundary[-1].reverse()
    for ellipse, matrix in zip(s_ellipse_boundary, s_matrix_boundary):
        ellipse_source_point = ellipse[0][1]
        matrix_source_point = matrix[0][1]
        gmshOCC.addLine(ellipse_source_point, matrix_source_point)
    t_matrix_boundary[-1].reverse()
    for ellipse, matrix in zip(t_ellipse_boundary, t_matrix_boundary):
        ellipse_target_point = ellipse[0][1]
        matrix_target_point = matrix[0][1]
        gmshOCC.addLine(ellipse_target_point, matrix_target_point)
        
    surf_ref_lst = curveLoopGenerator([curve_ref, curve_ref + 17, curve_ref + 8, curve_ref + 16], 
                       [False, True, False, False], 8, 
                       [False, False, False, False])
    
    #Butterfly prism (4 grouped surfaces):
    curves1 = gmshOCC.addCurveLoop([curve_ref + 63, curve_ref + 56, 
                                    curve_ref + 65, curve_ref + 48, 
                                    curve_ref + 55, curve_ref + 71])
    gmshOCC.addSurfaceFilling(curves1)
    curves2 = gmshOCC.addCurveLoop([curve_ref + 57, curve_ref + 58, 
                                    curve_ref + 67, curve_ref + 50, 
                                    curve_ref + 49, curve_ref + 65])
    gmshOCC.addSurfaceFilling(curves2)
    curves3 = gmshOCC.addCurveLoop([curve_ref + 59, curve_ref + 60, 
                                    curve_ref + 69, curve_ref + 52, 
                                    curve_ref + 51, curve_ref + 67])
    gmshOCC.addSurfaceFilling(curves3)
    curves4 = gmshOCC.addCurveLoop([curve_ref + 61, curve_ref + 62, 
                                    curve_ref + 71, curve_ref + 54, 
                                    curve_ref + 53, curve_ref + 69])
    gmshOCC.addSurfaceFilling(curves4)
    
    curveLoopGenerator([curve_ref + 48, curve_ref + 65, curve_ref + 56, curve_ref + 64], 
                       [False, True, False, False], 8, 
                       [False, False, False, False])
    
    curveLoopGenerator([curve_ref + 40, curve_ref + 72, curve_ref + 64, curve_ref + 80], 
                       [False, False, False, False], 8, 
                       [False, False, False, False])
       
    curveLoopGenerator([curve_ref + 88, curve_ref + 40, curve_ref + 96, curve_ref + 16], 
                       [False, False, False, False], 8, 
                       [False, False, False, False])
    
    curveLoopGenerator([curve_ref + 24, curve_ref + 41, curve_ref + 32, curve_ref + 40], 
                       [False, True, False, False], 8, 
                       [False, False, False, False])
    
    curveLoopGenerator([curve_ref + 8, curve_ref + 97, curve_ref + 32, curve_ref + 96], 
                       [False, True, False, False], 8, 
                       [False, False, False, False])
    
    curveLoopGenerator([curve_ref, curve_ref + 89, curve_ref + 24, curve_ref + 88], 
                       [False, True, False, False], 8, 
                       [False, False, False, False])
    
    curveLoopGenerator([curve_ref + 32, curve_ref + 81, curve_ref + 56, curve_ref + 80], 
                       [False, True, False, False], 8, 
                       [False, False, False, False])
   
    curveLoopGenerator([curve_ref + 24, curve_ref + 73, curve_ref + 48, curve_ref + 72], 
                       [False, True, False, False], 8, 
                       [False, False, False, False])
    
    curveLoopGenerator([curve_ref + 56, curve_ref + 57, curve_ref + 58, curve_ref + 59, 
                        curve_ref + 60, curve_ref + 61, curve_ref + 62, curve_ref + 63], 
                       [False, False, False, False, False, False, False, False], 1, 
                       [False, False, False, False, False, False, False, False])
    
    curveLoopGenerator([curve_ref + 48, curve_ref + 49, curve_ref + 50, curve_ref + 51,
                        curve_ref + 52, curve_ref + 53, curve_ref + 54, curve_ref + 55], 
                       [False, False, False, False, False, False, False, False], 1, 
                       [False, False, False, False, False, False, False, False])
    surf_ref = surf_ref_lst[0]
    
    vol_references = surfaceLoopGenerator([surf_ref, surf_ref + 29, surf_ref + 36, 
                          surf_ref + 28, surf_ref + 52 , surf_ref + 44], 
                         [False, True, False, False, False, False], 8, 
                         [False, False, False, False, False, False])
    vol_ref = vol_references[0]
    surfaceLoopGenerator([surf_ref + 36, surf_ref + 21, surf_ref + 12, 
                          surf_ref + 20, surf_ref + 68 , surf_ref + 60], 
                         [False, True, False, False, False, False], 8, 
                         [False, False, False, False, False, False])
    
    surfaceLoopGenerator([surf_ref + 8, surf_ref + 9, surf_ref + 10, 
                          surf_ref + 11, surf_ref + 76, surf_ref + 77], 
                         [False, False, False, False, False, False], 1, 
                         [False, False, False, False, False, False])
    gmshOCC.synchronize()
    
    #Transfinite Settings:
    #Useful tagging variables:
    maxLTag = gmshOCC.getMaxTag(1)
    maxSTag = gmshOCC.getMaxTag(2)
    maxVTag = gmshOCC.getMaxTag(3)
    #We want to be able to specify the number of nodes on certain types of curves.
    #In this model, there are two main types of curves: source/target face curves
    #and connecting, intermediate curves. To do obtain this curve data, we concatenate 
    #a list of all the curve dimensions that are in between the sourve and target faces:
    all_curves = gmshOCC.getEntities(1)
    #All of connecting curves were generated in the surface sewing process,
    #most of whom, were out of order, so we specify these curves manually in 'sewed_curves':
    sewed_curves = [curve_ref + 153, curve_ref + 157, curve_ref + 160, 
                    curve_ref + 164, curve_ref + 168, curve_ref + 172, 
                    curve_ref + 177, curve_ref + 181, curve_ref + 185, 
                    curve_ref + 190, curve_ref + 200, curve_ref + 205, 
                    curve_ref + 210, curve_ref + 215, curve_ref + 220,
                    curve_ref + 40, curve_ref + 41, curve_ref + 42,
                    curve_ref + 43, curve_ref + 44, curve_ref + 45, 
                    curve_ref + 46, curve_ref + 47, curve_ref + 104, 
                    curve_ref + 106, curve_ref + 108, curve_ref + 110,
                    curve_ref + 112, curve_ref + 114,curve_ref + 116,
                    curve_ref + 118, curve_ref + 195]
    connecting_curves = []
    for elem in all_curves:
        boundaries = gmshMOD.getBoundary([elem])
        if boundaries[1][1] - boundaries[0][1] == 8 or boundaries[1][1] - boundaries[0][1] == 16:
            connecting_curves.append(elem)
        elif elem[1] in sewed_curves:
            connecting_curves.append(elem)
    connecting_curves_tags = [elem[1] for elem in connecting_curves]
    #Based off of whether the curve is 'connecting' or 'source' and 'target',
    #we will define a transfinite curve with numNodes determined by this functions
    #'transfinite_curves' variable:
    s_and_t = transfinite_curves[0]
    inter = transfinite_curves[1]
    for i in range(curve_ref - 48, maxLTag + 1):
        if i in connecting_curves_tags:
            gmshMESH.setTransfiniteCurve(i, inter)
        else:
            gmshMESH.setTransfiniteCurve(i, s_and_t)
    #In defining transfinite surfaces, there are also special cases (mostly related
    #to the not well-defined ellipse regions) that we have to define the outer
    #pointTags for. Additionally, we will also use the corresponding point tag
    #reference variable, 'point_ref_tag' to describe the relevant point tags for
    #each matrix ID:
    for i in range(surf_ref, maxSTag + 1):
         if i == surf_ref + 76:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 59, 
                                                        point_ref_tag + 61, 
                                                        point_ref_tag + 63, 
                                                        point_ref_tag + 65])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref + 77:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 51, 
                                                        point_ref_tag + 53, 
                                                        point_ref_tag + 55, 
                                                        point_ref_tag + 57])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref + 8:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 65, 
                                                        point_ref_tag + 59, 
                                                        point_ref_tag + 51, 
                                                        point_ref_tag + 57])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref + 9:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 59, 
                                                        point_ref_tag + 61, 
                                                        point_ref_tag + 51, 
                                                        point_ref_tag + 53])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref + 10:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 61, 
                                                        point_ref_tag + 63, 
                                                        point_ref_tag + 53, 
                                                        point_ref_tag + 55])
             gmshMESH.setRecombine(2, i)
         elif i == surf_ref + 11:
             gmshMESH.setTransfiniteSurface(i, "Left", [point_ref_tag + 63, 
                                                        point_ref_tag + 65, 
                                                        point_ref_tag + 55, 
                                                        point_ref_tag + 57])
             gmshMESH.setRecombine(2, i)    
         else:
             gmshMESH.setTransfiniteSurface(i)
             gmshMESH.setRecombine(2, i)
    
    #There is only one special case in defining transfinite volumes for the central
    #butterfly region, which we define accordingly:
    for i in range(vol_ref, maxVTag + 1):
          if i == maxVTag:
              gmshMESH.setTransfiniteVolume(i, [point_ref_tag + 59, 
                                                point_ref_tag + 61, 
                                                point_ref_tag + 63, 
                                                point_ref_tag + 65, 
                                                point_ref_tag + 51, 
                                                point_ref_tag + 53, 
                                                point_ref_tag + 55, 
                                                point_ref_tag + 57])
          else:
              gmshMESH.setTransfiniteVolume(i)
    
    #We're done, nice! If you read this far, go outside and eat some grass -_-
    #don't actually eat grass
    all_entities = gmshOCC.getEntities()
    gmsh.model.occ.synchronize()
    
    return all_entities

def matrixfibergeo_edges(transfinite, matrix_ID = 0, angle = 30):
    #Main Dimensions of one Matrix-Fiber Layer:
    height = 11
    height_total = 11
    length = 6
    cent = 0.22  #fiber centerline distance
    t = 0.2  #matrix thickness
    ang = angle  #fiber orientation in deg

    #The width of one of these matrix bodies projected onto the most
    #bottom-most edge in the geometry in mm:
    width = cent/cos(ang*pi/180)

    #The height of one of these matrix bodies projected onto the most
    #right-most edge in the geometry in mm:
    height = cent/sin(ang*pi/180)

    #Corner offset, which avoids weird, unsymmetrical fiber geometries:
    off = 0.08

    #Number of fibers to the right of this corner offset matrix:
    N_fib = 22

    #The corner fibers in this model are always going to be weird and are edge cases
    #when defining a mesh for this geometry. Initially, it will be easier to ignore
    #these cases and start meshing the full matrix-fiber bodies themselves. Based off
    #of the SolidWorks parts, the origin (0, 0, 0) is defined as the bottom right hand
    #corner with the non-matrix-fiber body and edge with the 0.08 mm offset.

    #In dealing with this non-matrix-fiber body offset, the closest matrix body, which
    #is to the left of the defined origin (negative x-value), has some starting x
    #value for its right-most edge. After doing some trigonometry and geometry
    #breakdowns, this offset is defined as:
    start = length - width/2 - off - N_fib*width

    #Additionally, for geometry/mesh preparation, it will useful to identify these
    #matrix-fibers bodies by unique IDs. Starting off with the first offset matrix:
    mat_ID = matrix_ID
    
    #To simplify the function-calling process, gmshOCC and gmshMOD is defined:
    gmshOCC = gmsh.model.occ

    #Geometry Generation
    #The origin:
    origin = gmshOCC.addPoint(0, 0, 0)
    extr_ent = gmshOCC.extrude([(0, origin)], 0, t, 0)
    extr_pt = dimTagReturner(0, extr_ent)

    #Next, we wanna create the outer most boundaries of corner-most fiber. To do so,
    #we first define the source and target faces by starting at on of their midpoints.
    #This procedure will make more sense later on, when we define ellipse points.
    #Some key variables include:
    x_start = -start - mat_ID*width
    z_start = -abs(x_start)*tan((90 - ang)*pi/180)
 
    left_bot = gmshOCC.extrude([(0, origin)], x_start, 0, 0)
    up_bot = gmshOCC.extrude([(0, origin)], 0, 0, z_start)
    left_top = gmshOCC.extrude(extr_pt, x_start, 0, 0)
    up_top = gmshOCC.extrude(extr_pt, 0, 0, z_start)
    left_up = gmshOCC.addLine(dimTagReturner(0, left_bot)[0][1], dimTagReturner(0, left_top)[0][1])
    up_up = gmshOCC.addLine(dimTagReturner(0, up_bot)[0][1], dimTagReturner(0, up_top)[0][1])
    diag_bot = gmshOCC.addLine(dimTagReturner(0, left_bot)[0][1], dimTagReturner(0, up_bot)[0][1])
    diag_top = gmshOCC.addLine(dimTagReturner(0, left_top)[0][1], dimTagReturner(0, up_top)[0][1])
    
    front = gmshOCC.addSurfaceFilling(gmshOCC.addCurveLoop([dimTagReturner(1, extr_ent)[0][1], 
                                                    dimTagReturner(1, left_bot)[0][1], left_up, 
                                                    dimTagReturner(1, left_top)[0][1]]))
    right = gmshOCC.addSurfaceFilling(gmshOCC.addCurveLoop([dimTagReturner(1, extr_ent)[0][1], 
                                                    dimTagReturner(1, up_bot)[0][1], up_up, 
                                                    dimTagReturner(1, up_top)[0][1]]))
    back = gmshOCC.addSurfaceFilling(gmshOCC.addCurveLoop([left_up, diag_bot, up_up, diag_top]))
    top = gmshOCC.addSurfaceFilling(gmshOCC.addCurveLoop([dimTagReturner(1, left_top)[0][1], 
                                                    diag_top, dimTagReturner(1, up_top)[0][1]]))
    bot = gmshOCC.addSurfaceFilling(gmshOCC.addCurveLoop([dimTagReturner(1, left_bot)[0][1], 
                                                    diag_bot, dimTagReturner(1, up_bot)[0][1]]))
    vol1 = gmshOCC.addVolume([gmshOCC.addSurfaceLoop([front, right, back, top, bot])])
    gmshOCC.synchronize()
    
    #Transfinite Settings:
    maxLTag = gmshOCC.getMaxTag(1)
    maxSTag = gmshOCC.getMaxTag(2)
    for i in range(dimTagReturner(1, extr_ent)[0][1], maxLTag + 1):
        if i == dimTagReturner(1, extr_ent)[0][1] or i == left_up or i == up_up:
            gmshMESH.setTransfiniteCurve(i, transfinite[1])
        else:
            gmshMESH.setTransfiniteCurve(i, transfinite[0])
    for i in range(front, maxSTag + 1):
        gmshMESH.setTransfiniteSurface(i)
        gmshMESH.setRecombine(2, i)
    gmshMESH.setTransfiniteVolume(vol1)
    
    d1 = height_total - start*tan((90 - ang)*pi/180) - 24*height
    d2 = d1*sin(ang*pi/180)
    d3 = cent - d2
    off_left = d3/cos(ang*pi/180)
    start_left = length - off_left - N_fib*width
    x_start_left = -length + start_left + mat_ID*width
    z_start_left = (length + x_start_left)/tan((ang)*pi/180) - height_total
    
    opp_corner = gmshOCC.addPoint(-length, 0, -height_total)
    extr_ent_opp = gmshOCC.extrude([(0, opp_corner)], 0, t, 0)
    left_bot_opp = gmshOCC.extrude([(0, opp_corner)], 0, 0, height_total + z_start_left)
    left_top_opp = gmshOCC.extrude(dimTagReturner(0, extr_ent_opp), 0, 0, height_total + z_start_left)
    up_bot_opp = gmshOCC.extrude([(0, opp_corner)], length + x_start_left, 0, 0)
    up_top_opp = gmshOCC.extrude(dimTagReturner(0, extr_ent_opp), length + x_start_left, 0, 0)
    left_up_opp = gmshOCC.addLine(dimTagReturner(0, left_bot_opp)[0][1], dimTagReturner(0, left_top_opp)[0][1])
    up_up_opp = gmshOCC.addLine(dimTagReturner(0, up_bot_opp)[0][1], dimTagReturner(0, up_top_opp)[0][1])
    diag_bot_opp = gmshOCC.addLine(dimTagReturner(0, left_bot_opp)[0][1], dimTagReturner(0, up_bot_opp)[0][1])
    diag_top_opp = gmshOCC.addLine(dimTagReturner(0, left_top_opp)[0][1], dimTagReturner(0, up_top_opp)[0][1])
    front = gmshOCC.addSurfaceFilling(gmshOCC.addCurveLoop([dimTagReturner(1, extr_ent_opp)[0][1], 
                                                    dimTagReturner(1, left_bot_opp)[0][1], left_up_opp, 
                                                    dimTagReturner(1, left_top_opp)[0][1]]))
    right = gmshOCC.addSurfaceFilling(gmshOCC.addCurveLoop([dimTagReturner(1, extr_ent_opp)[0][1], 
                                                    dimTagReturner(1, up_bot_opp)[0][1], up_up_opp, 
                                                    dimTagReturner(1, up_top_opp)[0][1]]))
    back = gmshOCC.addSurfaceFilling(gmshOCC.addCurveLoop([left_up_opp, diag_bot_opp, up_up_opp, diag_top_opp]))
    top = gmshOCC.addSurfaceFilling(gmshOCC.addCurveLoop([dimTagReturner(1, left_top_opp)[0][1], 
                                                    diag_top_opp, dimTagReturner(1, up_top_opp)[0][1]]))
    bot = gmshOCC.addSurfaceFilling(gmshOCC.addCurveLoop([dimTagReturner(1, left_bot_opp)[0][1], 
                                                    diag_bot_opp, dimTagReturner(1, up_bot_opp)[0][1]]))
    vol2 = gmshOCC.addVolume([gmshOCC.addSurfaceLoop([front, right, back, top, bot])])
    gmshOCC.synchronize()
    
    #Transfinite Settings:
    maxLTag = gmshOCC.getMaxTag(1)
    maxSTag = gmshOCC.getMaxTag(2)
    for i in range(dimTagReturner(1, extr_ent_opp)[0][1], maxLTag + 1):
        if i == dimTagReturner(1, extr_ent_opp)[0][1] or i == left_up_opp or i == up_up_opp:
            gmshMESH.setTransfiniteCurve(i, transfinite[1])
        else:
            gmshMESH.setTransfiniteCurve(i, transfinite[0])
    for i in range(front, maxSTag + 1):
        gmshMESH.setTransfiniteSurface(i)
        gmshMESH.setRecombine(2, i)
    gmshMESH.setTransfiniteVolume(vol2)
    
    return None

def geometrygenerator(N):
    """
     geometrygenerator(N)

    Defines 'N' number of matrix-fiber bodies starting with matrix-ID 0

    Returns 'N' matrix-fiber geometry entities
    """
    all_all_entities = []
    for i in range(N):
        all_all_entities.append(matrixfibergeo_right(i, [4, 5]))
        all_all_entities.append(matrixfibergeo_left(i, [4, 5]))
    return all_all_entities

height = 11
length = 6 
cent = 0.22  #fiber centerline distance
d = 0.12  #fiber diameter
r = d/2  #fiber radius
t = 0.2  #matrix thickness
numBodies = 23

matrixfibergeo_central([4, 5])
matrixfibergeo_edges([5, 7])
geometrygenerator(numBodies)

#gmshOCC.addBox(0, 0, 0, -length, t, -height)

gmshOCC.synchronize()

#Mesh Generation
gmsh.option.setNumber("Mesh.Smoothing", 0)
#gmsh.model.mesh.generate(2)

#Comments
v = gmsh.view.add("comments")
gmsh.view.addListDataString(v, [10, -10], ["Created by Yousuf Abubakr"])
   
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

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

def matrixPrismGenertor(centers, xtranslation, ytranslations, ztranslation):
    """
     matrixPrismGenertor(centers, xtranslation, ytranslations, ztranslation)

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
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), -xtranslation, 0, 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), -xtranslation, 0, 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, -ytranslations[0], 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, -ytranslations[0], 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), xtranslation, 0, 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), xtranslation, 0, 0))
    extrusions.append(tuple([1, gmshOCC.addLine(gmshOCC.getMaxTag(0), center1_start)]))
    #Defining the target point and creating extrusions to create base for target face:
    center2_start = gmshOCC.addPoint(centers[1][0], centers[1][1], centers[1][2])
    extrusions.append(gmshOCC.extrude([(0, center2_start)], 0, ytranslations[1], 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, -ztranslation))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, -ztranslation))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, -ytranslations[1], 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, -ytranslations[1], 0))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, ztranslation))
    extrusions.append(gmshOCC.extrude(dimTagReturner(0, extrusions[-1]), 0, 0, ztranslation))
    extrusions.append(tuple([1, gmshOCC.addLine(gmshOCC.getMaxTag(0), center2_start)]))
    #Connecting the source and target base faces with lines:
    for i in range(8):
        gmshOCC.addLine(center1_start + i, center2_start + i)
    return extrusions

def curveLoopGenerator(curveTags, loopBools, iterNums, jumpBools, curveJump = 1, pointTags = []):
    """
     curveLoopGenerator(curveTags, loopBools, iterNums)

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
    iter = 0
    for elem in curveLoopLst:
        #print(list(elem))
        curves = gmshOCC.addCurveLoop(list(elem))
        if pointTags == []:
            gmshOCC.addSurfaceFilling(curves, -1)
        else:
            gmshOCC.addSurfaceFilling(curves, -1, [pointTags[0] + iter, pointTags[1] + iter])
            iter += 2
    #print("Curve Loops being defined: ", curveLoopLst)
    return curveLoopLst

def surfaceLoopGenerator(surfTags, loopBools, iterNums, jumpBools, surfJump = 1):
    """
     curveLoopGenerator(curveTags, loopBools, iterNums, jumpBools, surfJump = 1)

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
    for elem in surfLoopLst:
        #print(list(elem))
        surfs = gmshOCC.addSurfaceLoop(list(elem), -1, True)
        gmshOCC.addVolume([surfs])
    #print("Surface Loops being defined: ", surfLoopLst)
    return surfLoopLst

#Main Function:
def matrixfibergeo(matrix_ID, transfinite_curves, angle = 30, but1 = 1/3, but2 = 2/9):
    """
     matrixfibergeo(matrix_ID)

    Defines geometry for one matrix-fiber set that is based off of the SolidWorks
    Part files, starting with the matrix-fiber set with fibers angled cross-ply at some
    angle 'angle' with matrix ID, 'matrix_ID', such that a matrix ID of 0 corresponds 
    to the right-most, corner-most matrix-fiber body, and 'but1' and 'but2' represent 
    the butterfly ratios, which is the ratio determining how large the butterfly 
    prism will be with respect to the elliptical boundary, for the source and 
    target butterfly prisms. 'transfinite_curves' is a list of dim = 2 that contains
    numNodes data, such that the first element represents the number of nodes on the
    source and target face curves, while the second element represents the number of
    nodes on the curves in between the source and target faces.

    Returns matrix-fiber set geometry entities
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
    #The origin
    #origin = gmshOCC.addPoint(0, 0, 0)
    
    #Next, we wanna create the outer most boundaries of corner-most fiber. To do so,
    #we first define the source and target faces by starting at on of their midpoints.
    #This procedure will make more sense later on, when we define ellipse points.
    #Some key variables include:
    x_start = -start - mat_ID*width
    y_start = t/2
    z_start = -abs(x_start)*tan((90 - ang)*pi/180)
    #Next, we wull define the curve boundaries. To do so, we will primarily use the utility
    #function 'matrixPrismGenerator,' which fully defines the curves we wouls like
    #to make:
    matrix = matrixPrismGenertor([[x_start, y_start, 0], [0, y_start, z_start]], width/2, [t/2, t/2], height/2)
    
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
    gmshOCC.addPoint(center_source[0], center_source[1], center_source[2])
    gmshOCC.addPoint(center_target[0], center_target[1], center_target[2])
    #Next, we define source_ell and target_ell that stores the generated ellipse
    #curve data to be able to generate coneecting lines between source and
    #target face ellipses
    source_ell = []
    target_ell = []
    mat_ID_param = 32
    for i in range(8):
        source_ell.append(gmshOCC.addEllipse(x_ell_start, y_ell_start, 0, maj_axis_source, min_axis, -1, i*pi/4, (i + 1)*pi/4))
        #target_ell.append(gmshOCC.addEllipse(0, y_ell_start, z_ell_start, maj_axis_target, min_axis, -1, i*pi/4, (i + 1)*pi/4))
        #print("Ellipse arcs on source face: ", source_ell, "Ellipse arcs on target face: ", target_ell)
        #Because the addEllipse function does not allow you to define the orientation
        #in which the ellipse is defined, we must automatically rotate its curves
        #to define it for our model:
        #gmshOCC.rotate([(1, 26 + 2*i + mat_ID_param*mat_ID)], 0, y_ell_start, z_ell_start, 0, -1, 0, pi/2)
    mat_ID_param = 256
    for i in range(8):
        target_ell.append(gmshOCC.addEllipse(0, y_ell_start, z_ell_start, maj_axis_target, min_axis, -1, i*pi/4, (i + 1)*pi/4))
        #print("Ellipse arcs on source face: ", source_ell, "Ellipse arcs on target face: ", target_ell)
        #Because the addEllipse function does not allow you to define the orientation
        #in which the ellipse is defined, we must automatically rotate its curves
        #to define it for our model:
        gmshOCC.rotate([(1, 33 + i + mat_ID_param*mat_ID)], 0, y_ell_start, z_ell_start, 0, -1, 0, pi/2)
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
                                           square_source, [square_source, square_target], square_target)
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
    #defines the surfaces we would like to make. We also need to define some key
    #tagging related variables that account for the tag differences across each
    #matrix ID:
    mat_ID_surfparam = 154
    mat_ID_volparam = 16
    mat_ID_pointparam = 98
    mat_ID_param = 256
    curveLoopGenerator([mat_ID_param*mat_ID + 1, mat_ID_param*mat_ID + 17, mat_ID_param*mat_ID + 9, mat_ID_param*mat_ID + 18], 
                           [False, False, False, True], 8, [False, False, False, False])
    #The butterfly prism, ellipses, and intermediate surfaces require a similar 
    #procedure as well:
    #Butterfly prism:
    curveLoopGenerator([mat_ID_param*mat_ID + 49, mat_ID_param*mat_ID + 65, mat_ID_param*mat_ID + 57, mat_ID_param*mat_ID + 66], 
                            [False, False, False, True], 8, [False, False, False, False])
    curves4 = gmshOCC.addCurveLoop([56 + mat_ID_param*mat_ID, 49 + mat_ID_param*mat_ID, 66 + mat_ID_param*mat_ID, 57 + mat_ID_param*mat_ID, 64 + mat_ID_param*mat_ID, 72 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves4)
    curves1 = gmshOCC.addCurveLoop([50 + mat_ID_param*mat_ID, 51 + mat_ID_param*mat_ID, 68 + mat_ID_param*mat_ID, 59 + mat_ID_param*mat_ID, 58 + mat_ID_param*mat_ID, 66 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves1)
    curves2 = gmshOCC.addCurveLoop([52 + mat_ID_param*mat_ID, 53 + mat_ID_param*mat_ID, 70 + mat_ID_param*mat_ID, 61 + mat_ID_param*mat_ID, 60 + mat_ID_param*mat_ID, 68 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves2)
    curves3 = gmshOCC.addCurveLoop([54 + mat_ID_param*mat_ID, 55 + mat_ID_param*mat_ID, 72 + mat_ID_param*mat_ID, 63 + mat_ID_param*mat_ID, 62 + mat_ID_param*mat_ID, 70 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves3)
    #Connecting butterfly prism and ellipse:
    curveLoopGenerator([mat_ID_param*mat_ID + 73, mat_ID_param*mat_ID + 41, mat_ID_param*mat_ID + 81, mat_ID_param*mat_ID + 65], 
                            [False, False, False, False], 8, [False, False, False, False])
    #Connecting ellipse and matrix:
    curveLoopGenerator([mat_ID_param*mat_ID + 89, mat_ID_param*mat_ID + 17, mat_ID_param*mat_ID + 97, mat_ID_param*mat_ID + 41], 
                            [False, False, False, False], 8, [False, False, False, False])
    #Ellipse connecting faces:
    curveLoopGenerator([mat_ID_param*mat_ID + 25, mat_ID_param*mat_ID + 41, mat_ID_param*mat_ID + 33, mat_ID_param*mat_ID + 42], 
                            [False, False, False, True], 8, [False, False, False, False])
    
    #For practicality purposes, it is also useful to define the combined surface
    #of two faces, anf this will be helpful for the intermediate elliptical surfaces
    #later on. For these cases, it will be complicated to use the previously stated
    #function, so we manually define them:
    curves5 = gmshOCC.addCurveLoop([25 + mat_ID_param*mat_ID, 74 + mat_ID_param*mat_ID, 49 + mat_ID_param*mat_ID, 56 + mat_ID_param*mat_ID, 80 + mat_ID_param*mat_ID, 32 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves5)
    curves6 = gmshOCC.addCurveLoop([27 + mat_ID_param*mat_ID, 76 + mat_ID_param*mat_ID, 51 + mat_ID_param*mat_ID, 50 + mat_ID_param*mat_ID, 74 + mat_ID_param*mat_ID, 26 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves6)
    curves7 = gmshOCC.addCurveLoop([29 + mat_ID_param*mat_ID, 78 + mat_ID_param*mat_ID, 53 + mat_ID_param*mat_ID, 52 + mat_ID_param*mat_ID, 76 + mat_ID_param*mat_ID, 28 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves7)
    curves8 = gmshOCC.addCurveLoop([31 + mat_ID_param*mat_ID, 80 + mat_ID_param*mat_ID, 55 + mat_ID_param*mat_ID, 54 + mat_ID_param*mat_ID, 78 + mat_ID_param*mat_ID, 30 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves8)
    curves9 = gmshOCC.addCurveLoop([33 + mat_ID_param*mat_ID, 82 + mat_ID_param*mat_ID, 57 + mat_ID_param*mat_ID, 64 + mat_ID_param*mat_ID, 88 + mat_ID_param*mat_ID, 40 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves9)
    curves10 = gmshOCC.addCurveLoop([35 + mat_ID_param*mat_ID, 84 + mat_ID_param*mat_ID, 59 + mat_ID_param*mat_ID, 58 + mat_ID_param*mat_ID, 82 + mat_ID_param*mat_ID, 34 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves10)
    curves11 = gmshOCC.addCurveLoop([37 + mat_ID_param*mat_ID, 86 + mat_ID_param*mat_ID, 61 + mat_ID_param*mat_ID, 60 + mat_ID_param*mat_ID, 84 + mat_ID_param*mat_ID, 36 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves11)
    curves12 = gmshOCC.addCurveLoop([39 + mat_ID_param*mat_ID, 88 + mat_ID_param*mat_ID, 63 + mat_ID_param*mat_ID, 62 + mat_ID_param*mat_ID, 86 + mat_ID_param*mat_ID, 38 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves12)
    curves13 = gmshOCC.addCurveLoop([25 + mat_ID_param*mat_ID, 42 + mat_ID_param*mat_ID, 33 + mat_ID_param*mat_ID, 40 + mat_ID_param*mat_ID, 48 + mat_ID_param*mat_ID, 32 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves13)
    curves14 = gmshOCC.addCurveLoop([27 + mat_ID_param*mat_ID, 44 + mat_ID_param*mat_ID, 35 + mat_ID_param*mat_ID, 34 + mat_ID_param*mat_ID, 42 + mat_ID_param*mat_ID, 26 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves14)
    curves15 = gmshOCC.addCurveLoop([29 + mat_ID_param*mat_ID, 46 + mat_ID_param*mat_ID, 37 + mat_ID_param*mat_ID, 36 + mat_ID_param*mat_ID, 44 + mat_ID_param*mat_ID, 28 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves15)
    curves16 = gmshOCC.addCurveLoop([31 + mat_ID_param*mat_ID, 48 + mat_ID_param*mat_ID, 39 + mat_ID_param*mat_ID, 38 + mat_ID_param*mat_ID, 46 + mat_ID_param*mat_ID, 30 + mat_ID_param*mat_ID])
    gmshOCC.addSurfaceFilling(curves16)
    
    #Ellipse source faces:
    curveLoopGenerator([mat_ID_param*mat_ID + 25, mat_ID_param*mat_ID + 74, mat_ID_param*mat_ID + 49, mat_ID_param*mat_ID + 73], 
                            [False, True, False, False], 8, [False, False, False, False])
    #Ellipse target faces:
    curveLoopGenerator([mat_ID_param*mat_ID + 33, mat_ID_param*mat_ID + 82, mat_ID_param*mat_ID + 57, mat_ID_param*mat_ID + 81], 
                            [False, True, False, False], 8, [False, False, False, False])
    #Matrix source faces:
    curveLoopGenerator([mat_ID_param*mat_ID + 1, mat_ID_param*mat_ID + 90, mat_ID_param*mat_ID + 25, mat_ID_param*mat_ID + 89], 
                            [False, True, False, False], 8, [False, False, False, False])
    #Matrix target faces:
    curveLoopGenerator([mat_ID_param*mat_ID + 9, mat_ID_param*mat_ID + 98, mat_ID_param*mat_ID + 33, mat_ID_param*mat_ID + 97], 
                            [False, True, False, False], 8, [False, False, False, False])
    #Butterfly prism source faces:
    curveLoopGenerator([mat_ID_param*mat_ID + 49, mat_ID_param*mat_ID + 50, mat_ID_param*mat_ID + 51, mat_ID_param*mat_ID + 52, 
                        mat_ID_param*mat_ID + 53, mat_ID_param*mat_ID + 54, mat_ID_param*mat_ID + 55, mat_ID_param*mat_ID + 56], 
                            [False, False, False, False, False, False, False, False], 1, [False, False, False, False, False, False, False, False])
    #Butterfly prism target faces:
    curveLoopGenerator([mat_ID_param*mat_ID + 57, mat_ID_param*mat_ID + 58, mat_ID_param*mat_ID + 59, mat_ID_param*mat_ID + 60, 
                        mat_ID_param*mat_ID + 61, mat_ID_param*mat_ID + 62, mat_ID_param*mat_ID + 63, mat_ID_param*mat_ID + 64], 
                            [False, False, False, False, False, False, False, False], 1, [False, False, False, False, False, False, False, False])
   
    #Nice! We've now additionally defined all of our surfaces. Next up is surface
    #loops and volume definitions. To do so, we will primarily use the utility 
    #function 'surfaceLoopGenerator,' which fully defines the volumes we would 
    #like to make:
    #Outer matrix volume:
    surfaceLoopGenerator([mat_ID_surfparam*mat_ID + 1, mat_ID_surfparam*mat_ID + 30, mat_ID_surfparam*mat_ID + 37, mat_ID_surfparam*mat_ID + 29, 
                         mat_ID_surfparam*mat_ID + 73, mat_ID_surfparam*mat_ID + 81], 
                         [False, True, False, False, False, False], 8, [False, False, False, False, False, False])
    
    #Outer elliptical volume:
    surfaceLoopGenerator([mat_ID_surfparam*mat_ID + 37, mat_ID_surfparam*mat_ID + 22, mat_ID_surfparam*mat_ID + 9, mat_ID_surfparam*mat_ID + 21, 
                         mat_ID_surfparam*mat_ID + 57, mat_ID_surfparam*mat_ID + 65], 
                         [False, True, False, False, False, False], 8, [False, False, False, False, False, False])
    
    #Butterfly prism volume:
    surfaceLoopGenerator([mat_ID_surfparam*mat_ID + 17, mat_ID_surfparam*mat_ID + 18, mat_ID_surfparam*mat_ID + 19, mat_ID_surfparam*mat_ID + 20, 
                         mat_ID_surfparam*mat_ID + 89, mat_ID_surfparam*mat_ID + 90], 
                         [False, False, False, False, False, False], 1, [False, False, False, False, False, False])
    gmshOCC.synchronize()
     
    #Transfinite Settings:
    #Useful tagging variables:
    maxLTag = gmsh.model.occ.getMaxTag(1)
    maxSTag = gmsh.model.occ.getMaxTag(2)
    maxVTag = gmsh.model.occ.getMaxTag(3)
    #We want to be able to specify the number of nodes on certain types of curves.
    #In this model, there are two main types of curves: source/target face curves
    #and connecting, intermediate curves. To do obtain this curve data, we concatenate 
    #a list of all the curve dimensions that are in between the sourve and target faces:
    all_curves = gmshOCC.getEntities(1)
    #All of connecting curves were generated in the surface sewing process,
    #most of whom, were out of order, so we specify these curves manually in 'sewed_curves':
    sewed_curves = [mat_ID_param*mat_ID + 214, mat_ID_param*mat_ID + 219, mat_ID_param*mat_ID + 222, 
                    mat_ID_param*mat_ID + 227, mat_ID_param*mat_ID + 232, mat_ID_param*mat_ID + 237, 
                    mat_ID_param*mat_ID + 244, mat_ID_param*mat_ID + 249, mat_ID_param*mat_ID + 254, 
                    mat_ID_param*mat_ID + 209, mat_ID_param*mat_ID + 202, mat_ID_param*mat_ID + 197, 
                    mat_ID_param*mat_ID + 192, mat_ID_param*mat_ID + 187, mat_ID_param*mat_ID + 184, 
                    mat_ID_param*mat_ID + 179, mat_ID_param*mat_ID + 129, mat_ID_param*mat_ID + 133, 
                    mat_ID_param*mat_ID + 137, mat_ID_param*mat_ID + 141]
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
    for i in range(1 + mat_ID_param*mat_ID, maxLTag + 1):
        if i in connecting_curves_tags:
            gmshMESH.setTransfiniteCurve(i, inter)
        else:
            gmshMESH.setTransfiniteCurve(i, s_and_t)
    #In defining transfinite surfaces, there are also special cases (mostly related
    #to the not well-defined ellipse regions) that we have to define the outer
    #pointTags for:
    for i in range(1 + mat_ID_surfparam*mat_ID, maxSTag + 1):
         if i == mat_ID_surfparam*mat_ID + 53:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 33, 
                                                        mat_ID_pointparam*mat_ID + 20, 
                                                        mat_ID_pointparam*mat_ID + 37, 
                                                        mat_ID_pointparam*mat_ID + 49])
             gmshMESH.setRecombine(2, i)
         elif i == mat_ID_surfparam*mat_ID + 54:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 21, 
                                                        mat_ID_pointparam*mat_ID + 24, 
                                                        mat_ID_pointparam*mat_ID + 37, 
                                                        mat_ID_pointparam*mat_ID + 41])
             gmshMESH.setRecombine(2, i)
         elif i == mat_ID_surfparam*mat_ID + 55:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 25, 
                                                        mat_ID_pointparam*mat_ID + 28, 
                                                        mat_ID_pointparam*mat_ID + 41, 
                                                        mat_ID_pointparam*mat_ID + 45])
             gmshMESH.setRecombine(2, i)    
         elif i == mat_ID_surfparam*mat_ID + 56:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 29, 
                                                        mat_ID_pointparam*mat_ID + 32, 
                                                        mat_ID_pointparam*mat_ID + 45, 
                                                        mat_ID_pointparam*mat_ID + 49])
             gmshMESH.setRecombine(2, i)             
         elif i == mat_ID_surfparam*mat_ID + 45:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 20, 
                                                         mat_ID_pointparam*mat_ID + 33, 
                                                         mat_ID_pointparam*mat_ID + 52, 
                                                         mat_ID_pointparam*mat_ID + 58])
             gmshMESH.setRecombine(2, i)
         elif i == mat_ID_surfparam*mat_ID + 46:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 21, 
                                                        mat_ID_pointparam*mat_ID + 24, 
                                                        mat_ID_pointparam*mat_ID + 52, 
                                                        mat_ID_pointparam*mat_ID + 54])
             gmshMESH.setRecombine(2, i)
         elif i == mat_ID_surfparam*mat_ID + 47:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 25, 
                                                        mat_ID_pointparam*mat_ID + 28, 
                                                        mat_ID_pointparam*mat_ID + 54, 
                                                        mat_ID_pointparam*mat_ID + 56])
             gmshMESH.setRecombine(2, i)    
         elif i == mat_ID_surfparam*mat_ID + 48:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 29, 
                                                        mat_ID_pointparam*mat_ID + 32, 
                                                        mat_ID_pointparam*mat_ID + 56, 
                                                        mat_ID_pointparam*mat_ID + 58])
             gmshMESH.setRecombine(2, i)        
         elif i == mat_ID_surfparam*mat_ID + 49:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 36, 
                                                         mat_ID_pointparam*mat_ID + 49, 
                                                         mat_ID_pointparam*mat_ID + 60, 
                                                         mat_ID_pointparam*mat_ID + 66])
             gmshMESH.setRecombine(2, i)
         elif i == mat_ID_surfparam*mat_ID + 50:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 37, 
                                                        mat_ID_pointparam*mat_ID + 40, 
                                                        mat_ID_pointparam*mat_ID + 60, 
                                                        mat_ID_pointparam*mat_ID + 62])
             gmshMESH.setRecombine(2, i)
         elif i == mat_ID_surfparam*mat_ID + 51:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 41, 
                                                        mat_ID_pointparam*mat_ID + 44, 
                                                        mat_ID_pointparam*mat_ID + 62, 
                                                        mat_ID_pointparam*mat_ID + 64])
             gmshMESH.setRecombine(2, i)    
         elif i == mat_ID_surfparam*mat_ID + 52:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 45, 
                                                        mat_ID_pointparam*mat_ID + 48, 
                                                        mat_ID_pointparam*mat_ID + 64, 
                                                        mat_ID_pointparam*mat_ID + 66])
             gmshMESH.setRecombine(2, i)   
         elif i == mat_ID_surfparam*mat_ID + 89:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 52, 
                                                        mat_ID_pointparam*mat_ID + 54, 
                                                        mat_ID_pointparam*mat_ID + 56, 
                                                        mat_ID_pointparam*mat_ID + 58])
             gmshMESH.setRecombine(2, i)    
         elif i == mat_ID_surfparam*mat_ID + 90:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 60, 
                                                        mat_ID_pointparam*mat_ID + 62, 
                                                        mat_ID_pointparam*mat_ID + 64, 
                                                        mat_ID_pointparam*mat_ID + 66])
             gmshMESH.setRecombine(2, i)         
         elif i == mat_ID_surfparam*mat_ID + 17:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 52, 
                                                        mat_ID_pointparam*mat_ID + 58, 
                                                        mat_ID_pointparam*mat_ID + 60, 
                                                        mat_ID_pointparam*mat_ID + 66])
             gmshMESH.setRecombine(2, i)    
         elif i == mat_ID_surfparam*mat_ID + 18:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 52, 
                                                        mat_ID_pointparam*mat_ID + 54, 
                                                        mat_ID_pointparam*mat_ID + 60, 
                                                        mat_ID_pointparam*mat_ID + 62])
             gmshMESH.setRecombine(2, i)   
         elif i == mat_ID_surfparam*mat_ID + 19:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 54, 
                                                        mat_ID_pointparam*mat_ID + 56, 
                                                        mat_ID_pointparam*mat_ID + 62, 
                                                        mat_ID_pointparam*mat_ID + 64])
             gmshMESH.setRecombine(2, i)    
         elif i == mat_ID_surfparam*mat_ID + 20:
             gmshMESH.setTransfiniteSurface(i, "Left", [mat_ID_pointparam*mat_ID + 56, 
                                                        mat_ID_pointparam*mat_ID + 58, 
                                                        mat_ID_pointparam*mat_ID + 64, 
                                                        mat_ID_pointparam*mat_ID + 66])
             gmshMESH.setRecombine(2, i)

         else:
             gmshMESH.setTransfiniteSurface(i)
             gmshMESH.setRecombine(2, i) 
    #There is only one special case in defining transfinite volumes for the central
    #butterfly region, which we define accordingly:
    for i in range(1 + mat_ID_volparam*mat_ID, maxVTag + 1):
          if i == maxVTag:
              gmshMESH.setTransfiniteVolume(i, [mat_ID_pointparam*mat_ID + 52, 
                                                mat_ID_pointparam*mat_ID + 54, 
                                                mat_ID_pointparam*mat_ID + 56, 
                                                mat_ID_pointparam*mat_ID + 58, 
                                                mat_ID_pointparam*mat_ID + 60, 
                                                mat_ID_pointparam*mat_ID + 62, 
                                                mat_ID_pointparam*mat_ID + 64, 
                                                mat_ID_pointparam*mat_ID + 66])
          else:
              gmshMESH.setTransfiniteVolume(i)
    
    #We're done, nice! If you read this far, go outside and eat some grass -_-
    #don't actually eat grass
    all_entities = gmshOCC.getEntities()
    gmsh.model.occ.synchronize()
    
    return all_entities

def geometrygenerator(N):
    """
     geometrygenerator(N)

    Defines 'N' number of matrix-fiber bodies starting with matrix-ID 0

    Returns 'N' matrix-fiber geometry entities
    """
    all_all_entities = []
    for i in range(N):
        all_all_entities.append(matrixfibergeo(i, [7, 5]))
    return all_all_entities

numLayers = 1
cent = 0.22
width = cent/cos(30*pi/180)
length = 6
off = 0.08
N_fib = 22
start = length - width/2 - off - N_fib*width
mat_ID = numLayers - 1
x_start = -start - mat_ID*width
z_start = -abs(x_start)*tan((90 - 30)*pi/180)

geo = geometrygenerator(numLayers)

geo_copy = []
for elem in geo:
    geo_copy += gmshOCC.copy(elem)
gmshOCC.symmetrize(geo_copy, 1, 0, 0, abs(width) + abs(x_start))
gmshOCC.symmetrize(geo_copy, 0, 0, 1, 0)
gmshOCC.translate(geo_copy, 0, 0, z_start - 2*width)

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

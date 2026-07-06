#!/bin/sh
# The next line restarts using nmrWish \
exec nmrWish "$0" -notk -- "$@"

set auto_path    "[split $env(TCLPATH) :] $auto_path"
set ARGV         [concat $argv0 $argv]
set ARGC         [llength $ARGV]

set thisVersion "2020.311.09.12"

#
# 7/16/2013 corrections suggested by Brian Coggins and Pei Zhou of Duke University.

if {[flagLoc $ARGV -help]} \
   {
    puts "nusExpand.tcl Version $thisVersion"
    puts "Sort and Expand Non-Uniformly Sampled (NUS) Vectors:"
    puts " -ndim    ndim     \[Auto\]         Dimension Count of Output, 2-4."
    puts " -mode    inMode   \[pipe\]         Input Mode Keyword: pipe varian bruker."
    puts " -type    inType   \[Auto\]         Input Type Keyword: int float double Auto."
    puts " -prof    profName \[Auto\]         NMRPipe-Format NUS Kernel Output Name." 
    puts " -xSize   xSize    \[Auto\]         Vector Size in Input Points."
    puts " -qSize   qSize    \[Auto\]         Vector Quad Size, 1 = Real, 2 = Complex."
    puts " -qzSize  qzSize   \[Auto\]         Z-Axis Quad Size, 1 = Real, 2 = Complex."  
    puts " -qaSize  qaSize   \[Auto\]         A-Axis Quad Size, 1 = Real, 2 = Complex."  
    puts " -ySize   ySize    \[Auto\]         Number of xSize*qSize Vectors in Input."
    puts " -hw      hdrSize  \[Auto\]         Header Size in Input Words."
    puts " -pw      padSize  \[Auto\]         Pad Size in Input Words."
    puts " -skip    skipList \[None\]         Entries to Skip in Sample Schedule."
    puts " -defSign i        \[1\]            Default Sign for Output, 1 or -1."
    puts " -sign                            Invert Sign of Z-Axis/A-Axis Imaginaries."
    puts " -mask                            Output a Mask Instead of Data."
    puts " -expand                          Expand Data on Output (Default)."
    puts " -noexpand                        Input Data is Already Expanded."
    puts " -avg                             Redundant Increments are Averaged (Default)."
    puts " -noavg                           Redundant Increments are Summed (Also: -sum)."
    puts " -verb                            Verbose Mode ON (Default)."
    puts " -noverb                          Verbose Mode OFF."
    puts "Sample Schedule Format, First Item Location in Each Line is 0:"
    puts " -rev                             Reverse Items in Line Before Extraction."
    puts " -hasW                            Sample Schedule Includes Weights."
    puts " -yLoc        yLoc   \[0\]          Item Location of Y-Axis Sample." 
    puts " -zLoc        zLoc   \[1\]          Item Location of Z-Axis Sample." 
    puts " -aLoc        aLoc   \[2\]          Item Location of A-Axis Sample." 
    puts " -wLoc        wLoc   \[Auto\]       Item Location of Weights."
    puts " -sampleCount sCount \[Auto\]       Number of Valid Samples in Schedule."
    puts "Offset Subtracted from Schedule Values (Usually -1, 0, or 1):"
    puts " -offY        offY   \[0\]          Offset for Y-Axis Schedule Values."
    puts " -offZ        offZ   \[0\]          Offset for Z-Axis Schedule Values."
    puts " -offA        offA   \[0\]          Offset for A-Axis Schedule Values."
    puts " -off         oList  \[0\]          List of -offY -offZ -offA Values,"
    puts "                                  Or a Single Value to Set Offsets."
    puts "Use of Weighting Factors:"
    puts " -multW                           Multiply Spectral Data (pipe Mode Only)."
    puts " -profW                           Multiply NUS Schedule Output."
    puts "Sizes of Sorted, Expanded Output (in Complex Points if Dimension is Complex):"
    puts " -xT     xT       \[Auto\]          X-Axis Size of Output."
    puts " -yT     yT       \[Auto\]          Y-Axis Size of Output."
    puts " -zT     zT       \[Auto\]          Z-Axis Size of Output."
    puts " -aT     aT       \[Auto\]          A-Axis Size of Output."
    puts "Input Mode pipe:"
    puts " -in      inName  \[test.fid\]      NMRPipe-Format Input Name, or Keyword stdin."
    puts " -out     outName \[test_full.fid\] Adjusted Output Name, or Keyword stdout."
    puts " -type    inType  \[float\]         Input Type is Always float."
    puts " -sample  sName   \[Auto\]          Sample Schedule Input Name."
    puts " -hw      hSize   \[512\]           Header Size in Four-Byte Words."
    puts " -pw      pSize   \[0\]             Pad Size in Four-Byte Words."
    puts " -aqORD   aqORD   \[0\]             Quad Acquisition Order Code, 0 or 1."
    puts "Input Mode bruker:"
    puts " -in      inName  \[ser\]           Bruker ser Input Name, or Keyword stdin."
    puts " -out     outName \[ser_full\]      Adjusted ser Output Name, or Keyword stdout." 
    puts " -type    inType  \[Auto\]          Input Type Keyword: int float double Auto."
    puts " -sample  sName   \[Auto\]          Sample Schedule Input Name."
    puts " -hw      hSize   \[0\]             Header Size in Four-Byte Words."
    puts " -pw      pSize   \[0\]             Pad Size in Four-Byte Words."
    puts " -acqus   a1Name  \[acqus\]         Dimension 1 Parameter File."
    puts " -acqu2s  a2Name  \[acqu2s\]        Dimension 2 Parameter File."
#   puts "Input Mode nih:"
#   puts " -in      inName  \[ser\]           Bruker ser Input Name, or Keyword stdin."
#   puts " -out     outName \[ser_full\]      Adjusted ser Output Name, or Keyword stdout."
#   puts " -sample  sName   \[Auto\]          Sample Point Input Name."
#   puts " -acqus   a1Name  \[acqus\]         Dimension 1 Parameter File."
#   puts " -acqu2s  a2Name  \[acqu2s\]        Dimension 2 Parameter File."
    puts "Input Mode varian:"
    puts " -in      inName  \[fid\]           Varian-Format Input Name, or Keyword stdin."
    puts " -out     outName \[fid_full\]      Adjusted fid Output Name, or Keyword stdout."
    puts " -type    inType  \[Auto\]          Input Type Keyword: Auto int float"
    puts " -sample  sName   \[Auto\]          Sample Schedule Input Name."
    puts " -procpar pName   \[procpar\]       Input Parameter File."
    puts " -hw      hSize   \[8\]             Header Size in Four-Byte Words (Fixed)."
    puts " -pw      pSize   \[7\]             Pad Size in Four-Byte Words (Fixed)."
    puts "Notes:"
    puts " 1. The aqORD parameter sets the order of quad detection loops for"
    puts "    3D/4D Agilent/Varian data that has been converted to an intermediate"
    puts "    NMRPipe 2D format:"
    puts "    0 = 3D: zTime yTime zPhase yPhase"
    puts "        4D: aTime zTime yTime aPhase zPhase yPhase"
    puts "    1 = 3D: zTime yTime yPhase zPhase"
    puts "        4D: aTime zTime yTime yPhase zPhase aPhase"
    puts " 2. The -sign flag is also intended for 3D/4D Agilent/Varian"
    puts "    data that has been converted to an intermediate NMRPipe 2D format."
    puts " 3. If -hasW is used, weights are taken from the last item"
    puts "    in each line by default."
    puts " 4. Values for -skip locations start at 0. For example, this will skip"
    puts "    the first three entries in the sample schedule: -skip 0 1 2"
    exit 1
   }

set inMode         pipe
set inType         Auto 
set inName         test.fid
set sampleName     Auto 
set outName        Auto
set currentOutName stdout
set rpdName        None

set sampleCount  Auto
set skipList     None
set rpdCount     0
set rpdAdjust    1
set bytesPerWord 4
set hdrSize      Auto
set padSize      Auto 
set xSize        Auto
set qSize        Auto
set ySize        Auto
set locOffY      0
set locOffZ      0
set locOffA      0
set aqORD        0
set signFlag     0
set defSign      1
set multWFlag    0
set profWFlag    0
set maskFlag     0
set expandFlag   1
set rpdFlag      0
set realKey      1
set imagKey      2
set avgFlag      1

set yLoc         0 
set zLoc         1
set aLoc         2 
set wLoc         None 

getArgD $ARGV -mode   inMode
getArgD $ARGV -sample sampleName

getArgD $ARGV -hw     hdrSize
getArgD $ARGV -pw     padSize

if {![strcasecmp $sampleName Auto]} \
   {
    set sampleName [lindex [findNUSInput .] 0]
    if {![llength $sampleName]} {set sampleName None}
   }

if {![strcasecmp $inMode pipe]} \
   {
    if {![strcasecmp $hdrSize Auto]} {set hdrSize  $FDATASIZE}
    if {![strcasecmp $padSize Auto]} {set padSize 0}
   }

if {![strcasecmp $inMode varian]} \
   {
    set inName     fid 
    set outName    fid_full

    if {![strcasecmp $hdrSize Auto]}  {set hdrSize 8}
    if {![strcasecmp $padSize Auto]}  {set padSize 7}
   }

if {![strcasecmp $inMode bruker]} \
   {
    set inName     ser
    set outName    ser_full

    if {![strcasecmp $hdrSize Auto]} {set hdrSize 0}
    if {![strcasecmp $padSize Auto]} {set padSize 0}
   }

if {![strcasecmp $inMode nih]} \
   {
    set inName     ser
    set outName    ser_full

    if {![strcasecmp $hdrSize Auto]} {set hdrSize 0}
    if {![strcasecmp $padSize Auto]} {set padSize 0}
   }

set profName    Auto
set procparName procpar
set a1Name      acqus
set a2Name      acqu2s

set xT         Auto
set yT         Auto
set zT         Auto
set aT         Auto

set xtSize     0
set ytSize     0
set ztSize     0
set atSize     0

set qxSize     1
set qySize     Auto 
set qzSize     Auto
set qaSize     Auto

set dimCount   Auto

set revFlag    0
set verbose    1

getListArgD $ARGV -skip skipList

if {![strcasecmp $skipList None]} {set skipList ""}

getArgD $ARGV -in          inName
getArgD $ARGV -type        inType
getArgD $ARGV -out         outName
getArgD $ARGV -prof        profName
getArgD $ARGV -sampleCount sampleCount
getArgD $ARGV -rpd         rpdName
getArgD $ARGV -realKey     realKey
getArgD $ARGV -imagKey     imagKey 

getArgD $ARGV -procpar procparName
getArgD $ARGV -acqus   a1Name
getArgD $ARGV -acqu2s  a2Name

getArgD $ARGV -aqORD   aqORD
getArgD $ARGV -xSize   xSize
getArgD $ARGV -qSize   qSize
getArgD $ARGV -qySize  qySize
getArgD $ARGV -qzSize  qzSize
getArgD $ARGV -qaSize  qaSize
getArgD $ARGV -ySize   ySize

getArgD $ARGV -xT      xT
getArgD $ARGV -yT      yT
getArgD $ARGV -zT      zT
getArgD $ARGV -aT      aT

if {[flagLoc $ARGV -off]} \
   {
    set oList "$locOffY $locOffZ $locOffA"
    getListArgD $ARGV -off oList 

    set n [llength $oList]

    if {$n >= 1} \
       {
        set locOffY [lindex $oList 0]
        set locOffZ [lindex $oList 0]
        set locOffA [lindex $oList 0]
       }

    if {$n >= 2} \
       {
        set locOffZ [lindex $oList 1]
       }

    if {$n >= 3} \
       {
        set locOffA [lindex $oList 2]
       }
   }

getArgD $ARGV -offY locOffY
getArgD $ARGV -offZ locOffZ
getArgD $ARGV -offA locOffA

getArgD $ARGV -defSign defSign

if {[flagLoc $ARGV -expand]}   {set expandFlag 1}
if {[flagLoc $ARGV -noexpand]} {set expandFlag 0}

if {[flagLoc $ARGV -sign]}    {set signFlag 1}
if {[flagLoc $ARGV -nosign]}  {set signFlag 0}

if {[flagLoc $ARGV -mask]}    {set maskFlag 1}
if {[flagLoc $ARGV -nomask]}  {set maskFlag 0}

if {[flagLoc $ARGV -rev]}     {set revFlag 1}
if {[flagLoc $ARGV -norev]}   {set revFlag 0}

if {[flagLoc $ARGV -useW]}    {set wLoc Auto}
if {[flagLoc $ARGV -nouseW]}  {set wLoc None}

if {[flagLoc $ARGV -hasW]}    {set wLoc Auto}
if {[flagLoc $ARGV -nohasW]}  {set wLoc None}

if {[flagLoc $ARGV -multW]}   {set multWFlag 1}
if {[flagLoc $ARGV -nomultW]} {set multWFlag 0}

if {[flagLoc $ARGV -profW]}   {set profWFlag 1}
if {[flagLoc $ARGV -noprofW]} {set profWFlag 0}

if {[flagLoc $ARGV -avg]}     {set avgFlag 1}
if {[flagLoc $ARGV -noavg]}   {set avgFlag 0}
if {[flagLoc $ARGV -sum]}     {set avgFlag 0}

getArgD $ARGV -yLoc    yLoc
getArgD $ARGV -zLoc    zLoc
getArgD $ARGV -aLoc    aLoc
getArgD $ARGV -wLoc    wLoc

set loc [flagLoc $ARGV -verb]

if {$loc} \
   {
    set nextFlag [nextFlag $ARGV $loc]
    set verbose  1

    if {$nextFlag == $loc + 2} \
       {
        getArgD $ARGV -verb verbose
       }
   }

if {[flagLoc $ARGV -noverb]} {set verbose 0}

if {$hdrSize < 0} {set hdrSize 0}
if {$padSize < 0} {set padSize 0}

#
#--------------------------------------------------------------------------

proc getRPDSchedule {} \
{
   global rpdName rpdFlag realKey imagKey rpdCount rpdInfo

   if {![strcasecmp $rpdName None]} return

   set error [catch {set rUnit [open $rpdName r]}]

   if {$error} \
      {
       puts stderr "NUS Expand Error Opening RPD Input File $rpdName"
       exit 1
      }

   set n 0

   while {[gets $rUnit origLine] >= 0} \
      {
       set  line [string trim $origLine]
       incr n

       set itemCount [llength $line]
       if {!$itemCount} continue

       set key [string index [lindex $line 0] 0]
       if {![strcmp $key "#"]} continue

       if {![strcasecmp $key $realKey]} \
          {
           set rpdInfo($rpdCount) 1 
          } \
       elseif {![strcasecmp $key $imagKey]} \
          {
           set rpdInfo($rpdCount) 2 
          } \
       else \
          {
           puts stderr "NUS Expand Error: Unknown RPD Key $key in File $rpdName line $n."
           exit 1
          }

       incr rpdCount
      }

   set rpdFlag 1

   catch "close $rUnit"
}

proc getSampleParm {} \
{
   global inMode

   if {![strcasecmp $inMode pipe]} \
      {
       getSampleParmPipe
      } \
   elseif {![strcasecmp $inMode varian]} \
      {
       getSampleParmVarian
      } \
   elseif {![strcasecmp $inMode bruker] || ![strcasecmp $inMode NIH]} \
      {
       getSampleParmBruker
      } \
   else \
      {
       puts stderr "NUS Expand Error: Unsupported Mode $inMode."
       exit 1
      }

   showInputInfo
}

proc getSampleParmPipe {} \
{
    global xSize ySize qSize qxSize qySize qzSize qaSize xT yT zT aT aq2DMode hdrBuff 
    global xtSize ytSize ztSize atSize

    set qySizeOrig $qySize

    set qxSize 1
    set qySize 1

    set aq2DMode     [getParmI $hdrBuff FD2DPHASE  NULL_DIM]
    set thisDimCount [getParmI $hdrBuff FDDIMCOUNT NULL_DIM]

    if {![strcasecmp $xSize Auto]} {set xSize  [getParmI $hdrBuff NDSIZE CUR_XDIM]}
    if {![strcasecmp $ySize Auto]} {set ySize  [getParmI $hdrBuff NDSIZE CUR_YDIM]}

    set zSize  [getParmI $hdrBuff NDSIZE CUR_ZDIM]
    set aSize  [getParmI $hdrBuff NDSIZE CUR_ADIM]
    
    set xtSize [getParmI $hdrBuff NDAPOD CUR_XDIM]
    set ytSize [getParmI $hdrBuff NDAPOD CUR_YDIM]
    set ztSize [getParmI $hdrBuff NDAPOD CUR_ZDIM]
    set atSize [getParmI $hdrBuff NDAPOD CUR_ADIM]
    
    set xQuadSize [getQuadSize [getParmI $hdrBuff NDQUADFLAG CUR_XDIM]]
    set yQuadSize [getQuadSize [getParmI $hdrBuff NDQUADFLAG CUR_YDIM]]

    if {$thisDimCount >= 3 && ![strcasecmp $qzSize Auto]} {set qzSize [getQuadSize [getParmI $hdrBuff NDQUADFLAG CUR_ZDIM]]}
    if {$thisDimCount >= 4 && ![strcasecmp $qaSize Auto]} {set qaSize [getQuadSize [getParmI $hdrBuff NDQUADFLAG CUR_ADIM]]}

    if {$thisDimCount >= 3} {set ySize [expr $ySize*$zSize]}
    if {$thisDimCount >= 4} {set ySize [expr $ySize*$aSize]}

    if {$xQuadSize == 2 && $yQuadSize == 2} \
       {
        set qxSize 2
        set qySize 2

        if {![strcasecmp $qSize Auto]} {set qSize $qxSize}
        if {![strcasecmp $xT    Auto]} {set xT    $xSize}
       } \
    elseif {$xQuadSize == 1 && $yQuadSize == 1} \
       {
        set qxSize 1
        set qySize 1

        if {![strcasecmp $qSize Auto]} {set qSize $qxSize}
        if {![strcasecmp $xT    Auto]} {set xT    $xSize}
       } \
    else \
       {
        set qxSize 2
        set qySize 1

        if {![strcasecmp $qSize Auto]} {set qSize $qxSize}
        if {![strcasecmp $xT    Auto]} {set xT    $xSize}
       }

   if {[strcasecmp $qySizeOrig Auto]} {set qySize $qySizeOrig}
}

proc getSampleParmVarian {} \
{
    global inName xSize qSize ySize qySize xT procparName sampleName revFlag verbose

    set qySizeOrig $qySize

    if {![strcasecmp $xSize Auto]} \
       {
        if {![file exists $procparName]} \
           {
            puts stderr "NUS Expand Error Finding Parameter Input File $procparName"
            exit 1
           }

        set status [rdProcPar $procparName pList stderr]

        if {![info exists pList(np)]} \
           {
            puts stderr "NUS Expand Error Extracting np Parameter in File $procparName"
            exit 1
           }

        set xSize [expr $pList(np)/2]
       }

    if {![strcasecmp $ySize Auto]} \
       {
        set sInfo ""

        if {$revFlag} \
           {
            catch {set sInfo [exec getNUSInfo.tcl -sample $sampleName]}
           } \
        else \
           {
            catch {set sInfo [exec getNUSInfo.tcl -sample $sampleName -rev]}
           }

        if {![info exists sInfo]} {set sInfo ""}

        set n [getNamedItem vecCount $sInfo]

        if {[llength $n] && [string is integer $n]} {set ySize $n}
       }

    if {![strcasecmp $xSize Auto] || ![strcasecmp $ySize Auto]} \
       {
        puts stderr "NUS Expand Error Extracting Input Sizes from $procparName and $sampleName"
        exit 1
       }

    if {![strcasecmp $qSize Auto]} {set qSize 2}
    if {![strcasecmp $xT    Auto]} {set xT $xSize}

   if {[strcasecmp $qySizeOrig Auto]} {set qySize $qySizeOrig}
}

proc getSampleParmBruker {} \
{
    global xSize qSize ySize qySize xT acqus acqu2s sampleName revFlag verbose
    global bytesPerWord inType

    set qySizeOrig $qySize
    set qySize     2

    if {$bytesPerWord == 8} \
       {
        set serPadSize 128
       } \
    else \
       {
        set serPadSize 256
       }

    if {![strcasecmp $xSize Auto]} \
       {
        if {[info exists acqus(TD)]}  {set xSize [expr int($acqus(TD)/2)]}

        if {[info exists acqus(TD)]} \
           {
            set xSize [expr int($acqus(TD))]

            if {[expr $xSize % $serPadSize] > 0} \
               {
                set xSize [expr (int($xSize/$serPadSize)+1)*$serPadSize]  
               }

           set xSize [expr $xSize/2]
          }
      }

#   if {![strcasecmp $ySize Auto] && [info exists acqu2s(TD)]} {set ySize [expr int($acqu2s(TD))]}

    if {![strcasecmp $ySize Auto]} \
       {
        set sInfo "" 

        if {$revFlag} \
           {
            catch {set sInfo [exec getNUSInfo.tcl -sample $sampleName]}
           } \
        else \
           {
            catch {set sInfo [exec getNUSInfo.tcl -sample $sampleName -rev]}
           }

        if {![info exists sInfo]} {set sInfo ""}

        set n [getNamedItem vecCount $sInfo]

        if {[llength $n] && [string is integer $n]} {set ySize $n}
       }

    if {![strcasecmp $xSize Auto] || ![strcasecmp $ySize Auto]} \
       {
        puts stderr "NUS Expand Error Extracting Input Sizes from $a1Name and $a2Name"
        exit 1
       }

    if {![strcasecmp $qSize Auto]} {set qSize 2}
    if {![strcasecmp $xT    Auto]} {set xT $xSize}

    if {[strcasecmp $qySizeOrig Auto]} {set qySize $qySizeOrig}
}

proc showInputInfo {} \
{ 
    global inName inMode inType sampleName rpdName xSize ySize qSize hdrSize padSize bytesPerWord
    global rpdFlag expandFlag avgFlag verbose

    if {!$verbose} return

    set tStr Complex
    if {$qSize == 1} {set tStr Real}

    if {$expandFlag} \
       {
        set expandStr Yes
       } \
    else \
       {
        set expandStr No
       }

    if {$avgFlag} \
       {
        set avgStr Averaged
       } \
    else \
       {
        set avgStr Summed
       }

    puts stderr "NUS Expand. Input Mode:           $inMode"
    puts stderr "NUS Expand. Input Data:           $inName"
    puts stderr "NUS Expand. Input Type:           $inType"
    puts stderr "NUS Expand. Input Word Size:      $bytesPerWord"
    puts stderr "NUS Expand. Expand Input:         $expandStr"
    puts stderr "NUS Expand. Redundant Increments: $avgStr"
    puts stderr "NUS Expand. Input Vector Size:    $xSize $tStr Points"
    puts stderr "NUS Expand. Input Vector Count:   $ySize Vectors"

    if {[strcasecmp $inName stdin] && [file exists $inName]} \
       {
        set thisSize [file size $inName]
        set predSize [expr ($hdrSize + $padSize*$ySize + $xSize*$qSize*$ySize)*$bytesPerWord]

        puts stderr "NUS Expand. Size of Input:        $thisSize bytes"
        puts stderr "NUS Expand. Expected Size:        $predSize bytes"
       }

    puts stderr "NUS Expand. Input Schedule:       $sampleName"

    if {$rpdFlag} \
       {
        puts stderr "NUS Expand. RPD Schedule:         $rpdName"
       }
}

#
# Interpret sample scheme:

proc getSampleSchedule { } \
{
   global yLoc zLoc aLoc wLoc xSize ySize ytSize ztSize atSize sList wList dimCount verbose
   global locOffY locOffZ locOffA qxSize qySize qzSize qaSize xT yT zT aT wMin wMax revFlag
   global inName outName sampleName profName inMode expandFlag rpdName rpdFlag rpdCount rpdInfo
   global sampleCount maskFlag skipList

   set lineList ""
   set valList  ""
   set sList    ""
   set wList    ""

   set nusDim 2

   if {![strcasecmp $sampleName None]} \
      {
       if {![strcasecmp $dimCount Auto]} \
          {
           if {$verbose} {puts stderr "NUS Expand. No Sampling Schedule, Assuming 2D Data."}
           set dimCount 2
          }

       if {![strcasecmp $xT Auto]} {set xT $xSize}
       if {![strcasecmp $yT Auto]} {set yT $ySize}

       if {![strcasecmp $zT Auto]} {set zT 1}
       if {![strcasecmp $aT Auto]} {set aT 1}

       if {![strcasecmp $qzSize Auto]} {set qzSize 1}
       if {![strcasecmp $qaSize Auto]} {set qaSize 1}

       set nusDim [expr $dimCount - 1]

       for {set ia 0} {$ia < $aT} {incr ia} \
          {
           for {set iz 0} {$iz < $zT} {incr iz} \
              {
               for {set iy 0} {$iy < $yT} {incr iy} \
                  {
                   set iList [join [lrange "$iy $iz $ia" 0 [expr $nusDim - 1]] ","]
                   lappend sList $iList
                   lappend wList 1.0
                  }
              }
          }

       set wMin 1.0
       set wMax 1.0

       return
      }

   if {![file exists $sampleName]} \
      {
       puts stderr "NUS Expand Notice: No Sampling Input File."
       exit 1 
      }

   set skipCount  0
   set entryCount 0

   set n     0
   set str   ""

   set mxSize 0
   set mySize 0
   set mzSize 0
   set maSize 0
  
   set error [catch {set sUnit [open $sampleName r]}]

   if {$error} \
      {
       puts stderr "NUS Expand Error Opening Sampling Input File $sampleName"
       exit 1 
      }

   while {[gets $sUnit origLine] >= 0} \
      {
       set  line [string trim $origLine]
       incr n

       set itemCount [llength $line]
       if {!$itemCount} continue

       set key [string index [lindex $line 0] 0]
       if {![strcmp $key "#"]} continue

       set lineInfo($entryCount) $line

       incr entryCount 

       if {$entryCount == 1} \
          {
           set nD $itemCount

           if {![strcasecmp $wLoc Auto]} {set wLoc [expr $itemCount - 1]}
           if {[strcasecmp $wLoc None]}  {incr nD -1}

           if {![strcasecmp $dimCount Auto]} \
              {
               if {$nD == 2} \
                  {
                   set dimCount 3
                  } \
               elseif {$nD == 3} \
                  {
                   set dimCount 4
                  } \
               else \
                  {
                   set dimCount 2
                  }

               if {$verbose} {puts stderr "NUS Expand. Output Dimensions:    $dimCount"}
              }
          }

       if {$revFlag} {set line [lreverse [string trim $line]]}

       foreach val $line \
          {
           lappend valList  $val
           lappend lineList $n
          }
      }

    if {$verbose && $skipCount} {puts stderr ""}   
             
    catch "close $sUnit"

    if {[lsearch -exact "1 2 3 4" $dimCount] < 0} \
       {
        puts stderr "NUS Expand Error: Bad Dimension Count."
        exit 1 
       }

    set nV [llength $valList]
    set nI [expr $dimCount - 1]

    if {[strcasecmp $wLoc None]} {incr nI}

    if {$nV % $nI} \
       {
        puts stderr "NUS Expand Error: Number of Sample Values is Not a Multiple of $nI."
        exit 1 
       }

    set nCurrent [expr $nV/$nI]

    if {![strcasecmp $sampleCount Auto]} \
       {
        set sampleCount $nCurrent
       } \
    else \
       {
        if {$nCurrent > $sampleCount} \
           {
            set nV [expr $sampleCount*$nI]

            if {$verbose} \
               {
                puts stderr "NUS Expand. Sample Count:         Initial: $nCurrent Final: $sampleCount"
               }
           } \
        elseif {$nCurrent < $sampleCount} \
           {
            puts stderr ""
            puts stderr "NUS Expand. Warning: Requested Sample Count is too large:"
            puts stderr "NUS Expand. Initial: $nCurrent Requested: $sampleCount"
            puts stderr ""
           }
       }

    if {$rpdFlag && $rpdCount < $sampleCount} \
       {
        puts stderr ""
        puts stderr "NUS Expand. Error: too few values in RPD schedule."
        puts stderr "NUS Expand. RPD Values: $rpdCount NUS Samples: $sampleCount"
        puts stderr ""
        exit 1
       }

    set nusDim [expr $dimCount - 1] 
    set jList  ""
    set i 0

    if {$nusDim >= 1} {lappend jList $yLoc}
    if {$nusDim >= 2} {lappend jList $zLoc}
    if {$nusDim >= 3} {lappend jList $aLoc}
   
    set nItems 0

    while {$i < $nV} \
       {    
        set k [expr $i + $nI - 1]

        set iList [lrange $valList  $i $k]
        set nList [lrange $lineList $i $k]

        set vList ""
        set w     1.0
        set jj    0

        foreach j $jList \
           {
            set item [lindex $iList $j]
           
            if {![string is integer $item]} \
               {
                puts stderr "NUS Expand Error: Bad Sample Value Near Line [lindex $nList $j]: $item"
                exit 1
               } \
            else \
               {
                if {$jj == 2} \
                   {
                    set item [expr $item - $locOffA]
                    if {$item >= 0 && $item > $maSize} {set maSize $item}
                   } \
                elseif {$jj == 1} \
                   {
                    set item [expr $item - $locOffZ]
                    if {$item >= 0 && $item > $mzSize} {set mzSize $item}
                   } \
                else \
                   {
                    set item [expr $item - $locOffY]
                    if {$item >= 0 && $item > $mySize} {set mySize $item}
                   }
               }

            lappend vList $item
            incr jj
           }

        if {[strcasecmp $wLoc None]} \
           {
            set w [lindex $iList $wLoc]

            if {![string is double $w]} \
               {
                puts stderr "NUS Expand Error: Bad Weight Value Near Line [lindex $nList $j]: $w"
                exit 1
               }
           }

        if {[lsearch -exact $skipList $nItems] >= 0} \
           {
            if {$verbose} \
               {
                if {!$skipCount} {puts stderr ""}
                if {![info exists lineInfo($nItems)]} {set lineInfo($nItems) None}

                puts stderr "NUS Expand. Skipping Entry: [format %4s $nItems] Contents: $lineInfo($nItems)"
               }

            set w 0.0

            incr skipCount
           }

        if {!$nItems} \
           {
            set wMin $w
            set wMax $w
           }

        if {$w < $wMin} {set wMin $w}
        if {$w > $wMax} {set wMax $w}

        lappend wList $w
        lappend sList [join $vList ","]

        set i [expr $k + 1]
        incr nItems
       }

   if {$verbose && [llength $skipList]} {puts stderr ""}

   incr mySize
   incr mzSize
   incr maSize

   set sSize [llength $sList]

   if {![strcasecmp $inMode pipe]} \
      {
       if {![strcasecmp $xT Auto]} {set xT $xSize}

       if {![strcasecmp $yT Auto]} \
          {
           set yT $mySize
           if {$ytSize > $yT} {set yT $ytSize}
          }

#      if {$qxSize == 2 && $qySize == 2} {set yT [expr $yT/2]}
      } \
   else \
      {
       if {![strcasecmp $xT Auto]} {set xT $xSize}
       if {![strcasecmp $yT Auto]} {set yT $mySize}

       if {$dimCount >= 1} {set qxSize 2}

       if {$dimCount >= 2 && ![strcasecmp $qySize Auto]} {set qySize 2}
      }

   if {$dimCount >= 3 && ![strcasecmp $qzSize Auto]} {set qzSize 2}
   if {$dimCount >= 4 && ![strcasecmp $qaSize Auto]} {set qaSize 2}

   if {![strcasecmp $qzSize Auto]} {set qzSize 1}
   if {![strcasecmp $qaSize Auto]} {set qaSize 1}

   if {![strcasecmp $zT Auto]} \
      {
       set zT $mzSize
       if {$ztSize > $zT} {set zT $ztSize}
      }

   if {![strcasecmp $aT Auto]} \
      {
       set aT $maSize
       if {$atSize > $aT} {set aT $atSize}
      }

   if {$xT < 1} {set xT 1}
   if {$yT < 1} {set yT 1}
   if {$zT < 1} {set zT 1}
   if {$aT < 1} {set aT 1}

   if {$dimCount < 4} {set aT 1}
   if {$dimCount < 3} {set zT 1}
   if {$dimCount < 2} {set yT 1}

   set realCount 0
   set imagCount 0

   if {$rpdFlag} \
      {
       for {set i 0} {$i < $sampleCount} {incr i} \
          {
           if {[info exists rpdInfo($i)]} \
              {
               if {$rpdInfo($i) == 1} \
                  {
                   incr realCount
                  } \
               elseif {$rpdInfo($i) == 2} \
                  {
                   incr imagCount
                  }
              } \
           else \
              {
               set rpdInfo($i) 3
              } \
          }
      }
 
   set thisList "$xSize $qxSize $yT $qySize $zT $qzSize $aT $qaSize"

   if {$verbose} \
      {
       if {$maskFlag} \
          {
           set maskStr Yes
          } \
       else \
          {
           set maskStr No 
          }

       puts stderr "NUS Expand. Output Data:          $outName"
       puts stderr "NUS Expand. Mask Output:          $maskStr"
       puts stderr "NUS Expand. Output Kernel:        $profName"
       puts stderr "NUS Expand. NUS Samples Found:    $nCurrent"
       puts stderr "NUS Expand. NUS Samples Used:     $sSize"

       if {$rpdCount} \
          {
           set nS [expr $realCount + $imagCount]
           set b  0.0%

           if {$nS} {set b [format %.1f%% [expr 100.0 - 100.0*double($realCount - $imagCount)/double($nS)]]}
 
           puts stderr "NUS Expand. RPD Samples Found:    $rpdCount"
           puts stderr "NUS Expand. RPD Y-Axis Samples:   Real: $realCount Imaginary: $imagCount"
           puts stderr "NUS Expand. RPD Y-Axis Balance:   $b"
          }

       if {$dimCount == 2} \
          {
           puts stderr "NUS Expand. Index Offset:         Y: $locOffY"
          } \
       elseif {$dimCount == 3} \
          {
           puts stderr "NUS Expand. Index Offsets:        Y: $locOffY Z: $locOffZ"
          } \
       else \
          {
           puts stderr "NUS Expand. Index Offsets:        Y: $locOffY Z: $locOffZ A: $locOffA"
          }

       set fmt "%d %d"

       set nS [expr $yT*$zT*$aT]

       if {$dimCount >= 1} {puts stderr "NUS Expand. Output X-Size:        [format $fmt $xSize $qxSize]"}
       if {$dimCount >= 2} {puts stderr "NUS Expand. Output Y-Size:        [format $fmt $yT    $qySize]"}
       if {$dimCount >= 3} {puts stderr "NUS Expand. Output Z-Size:        [format $fmt $zT    $qzSize]"}
       if {$dimCount >= 4} {puts stderr "NUS Expand. Output A-Size:        [format $fmt $aT    $qaSize]"}

       puts stderr "NUS Expand. Output Increments:    [format %d $nS]"
       puts stderr "NUS Expand. Output Data Bytes:    [getSafeDataSize $thisList]"

       set nU  [listUniqCount $sList]
       set pct "0.0%"

       if {$nS} \
          {
           if {$rpdFlag} \
              {
               set pctStr [format "%.1f%% (RPD Adjusted)" [expr 0.5*100.0*$nU/$nS]]
              } \
           else \
              {
               set pctStr [format "%.1f%%" [expr 100.0*$nU/$nS]]
              }
          }

       puts stderr "NUS Expand. Sampling Density:     $pctStr"
      }

   if {![dataSizeIsSafe $thisList]} \
      {
       puts stderr ""
       puts stderr "NUS Expand. *** Warning: Data Size Might Exceed 2GB."
       puts stderr "NUS Expand. *** A 64-bit Version of NMRPipe Might Be Required."
      }
}

proc setLocInfo {} \
{
   global yT zT aT yInfo zInfo aInfo locInfo rpdInfo sList sampleCount dimCount

   set nusDim [expr $dimCount - 1] 
 
   for {set ia 0} {$ia < $aT} {incr ia} \
      {
       for {set iz 0} {$iz < $zT} {incr iz} \
          {
           set yInfo($ia,$iz) ""
          }
      }
  
   for {set ia 0} {$ia < $aT} {incr ia} \
      {
       for {set iy 0} {$iy < $yT} {incr iy} \
          {
           set zInfo($ia,$iy) ""
          }
      }

   for {set iz 0} {$iz < $zT} {incr iz} \
      {
       for {set iy 0} {$iy < $yT} {incr iy} \
          {
           set aInfo($iz,$iy) ""
          }
      }

   set iy 0
   set iz 0
   set ia 0

   foreach s $sList \
      {
       set  iList [split $s ","]

       if {$nusDim > 0} {set iy [lindex $iList 0]}
       if {$nusDim > 1} {set iz [lindex $iList 1]}
       if {$nusDim > 2} {set ia [lindex $iList 2]}

       if {$iy >= 0} {lappend yInfo($ia,$iz) $iy}
       if {$iz >= 0} {lappend zInfo($ia,$iy) $iz}
       if {$ia >= 0} {lappend aInfo($iz,$iy) $ia}
      }

   set i 0

   for {set ia 0} {$ia < $aT} {incr ia} \
      {
       for {set iz 0} {$iz < $zT} {incr iz} \
          {
           for {set iy 0} {$iy < $yT} {incr iy} \
              {
               set locInfo($i) "$iy $iz $ia"
               incr i
              }
          }
      }

   return 
}

proc getNUSOffset { ia iaq iz izq iy iyq loc } \
{
   global xSize xT yT zT aT qxSize qySize qzSize qaSize locInfo aqORD dimCount expandFlag

   if {$expandFlag} \
      {
       set off [expr $loc*$xSize*$qxSize*$qySize*$qzSize*$qaSize]
      } \
   else \
      {
       set off [expr ($iy + $iz*$yT + $ia*$yT*$zT)*$xSize*$qxSize*$qySize*$qzSize*$qaSize]
      }

   if {$aqORD == 0} \
      {
       if {$dimCount == 2} \
          {
           if {$iyq == 1} {incr off [expr $xSize*$qxSize]}
          } \
       elseif {$dimCount == 3} \
          {
           if {$izq == 1} {incr off [expr $xSize*$qxSize*$qySize]}
           if {$iyq == 1} {incr off [expr $xSize*$qxSize]}
          } \
       elseif {$dimCount == 4} \
          {
           if {$iaq == 1} {incr off [expr $xSize*$qxSize*$qySize*$qzSize]}
           if {$izq == 1} {incr off [expr $xSize*$qxSize*$qySize]}
           if {$iyq == 1} {incr off [expr $xSize*$qxSize]}
          }
      } \
   else \
      {
       if {$dimCount == 2} \
          {
           if {$iyq == 1} {incr off [expr $xSize*$qxSize]}
          } \
       elseif {$dimCount == 3} \
          {
           if {$iyq == 1} {incr off [expr $xSize*$qxSize*$qzSize]}
           if {$izq == 1} {incr off [expr $xSize*$qxSize]}
          } \
       elseif {$dimCount == 4} \
          {
           if {$iyq == 1} {incr off [expr $xSize*$qxSize*$qzSize*$qaSize]}
           if {$izq == 1} {incr off [expr $xSize*$qxSize*$qaSize]}
           if {$iaq == 1} {incr off [expr $xSize*$qxSize]}
          }
      }

   return $off
}

proc initInput {} \
{
   global inName inUnit

   if {![strcasecmp $inName stdin]} \
      {
       set inUnit 0
      } \
   else \
      {
       if {![file exists $inName]} \
          {
           puts stderr "NUS Expand Error Finding Input File $inName"
           exit 1
          }

       set inUnit [sysOpen $inName r]
      }
}

proc rdInput { thisUnit thisBuff n } \
{
   if {$n < 1 || ![strcmp $thisBuff 0]} {return 0}

   if {$thisUnit} \
      {
       set status [sysRead  $thisUnit $thisBuff $n]
      } \
   else \
      {
       set status [sysReadB $thisUnit $thisBuff $n]
      }

   return $status
}

proc wrOutput { thisUnit thisBuff n } \
{
   if {$n < 1 || ![strcmp $thisBuff 0] || ![strcasecmp $thisUnit None]} {return 0}
   
   set status [sysWrite $thisUnit $thisBuff $n]

   return $status
}

proc createPath { thisName } \
{
   set dName [file dirname $thisName]
   set error 0

   if {![file isdirectory $dName]} \
      {
       set error [catch "file mkdir $dName"]
      }

   return error
}

proc setOutputHdr {} \
{
   global inMode

   if {![strcasecmp $inMode pipe]} \
      {
       setOutputHdrPipe
      } \
   elseif {![strcasecmp $inMode varian]} \
      {
       setOutputHdrVarian
      }
}

proc setOutputHdrVarian {} \
{
   global hdrBuff hdrSize yT zT aT qySize qzSize qaSize dimCount

   if {!$hdrSize} return

   set fidCount 1

   if {$dimCount >= 2} {set fidCount [expr $fidCount*$yT*$qySize]}
   if {$dimCount >= 3} {set fidCount [expr $fidCount*$zT*$qzSize]}
   if {$dimCount >= 4} {set fidCount [expr $fidCount*$aT*$qaSize]}

   setVHdrInfo $hdrBuff "FIDCount $fidCount"
}

proc setOutputHdrPipe {} \
{
   global outName inBuff hdrBuff yT zT aT qxSize qySize qzSize qaSize aq2DMode dimCount
   global xSize ySize qSize nTotal maskFlag

   if {![strcasecmp $outName Auto]} \
      {
       set outName test_full.fid

       if {$dimCount == 3} {set outName fid_full/test%03d.fid}
       if {$dimCount == 4} {set outName fid_full/test%03d.fid}
      }

   set fmtCount  [getFmtCount $outName]
   set pipeFlag  0
   set cubeFlag  0
   set fileCount 1

   set minVal    0.0
   set maxVal    0.0

#  vSize  [expr $qSize*$xSize]
#  nTotal [expr $vSize*$ySize]

   if {$maskFlag} \
      {
       set minVal 0.0
       set maxVal 1.0
      } \
   else \
      {
       set jump [expr $qSize*$xSize]
       set rPtr $inBuff
       set off  0

       for {set iy 0} {$iy < $ySize} {incr iy} \
          {
           set thisMin [vMin $rPtr $xSize] 
           set thisMax [vMax $rPtr $xSize] 

           if {!$iy} \
              {
               set minVal $thisMin
               set maxVal $thisMax
              }

           if {$thisMax > $maxVal} {set maxVal $thisMax}
           if {$thisMin < $minVal} {set minVal $thisMin}

           set rPtr [incrPtr $rPtr $jump]
          }

       setParm $hdrBuff FDMIN       $minVal NULL_DIM
       setParm $hdrBuff FDMAX       $maxVal NULL_DIM
       setParm $hdrBuff FDSCALEFLAG 1.0     NULL_DIM
      }

   if {$dimCount == 4 && $fmtCount == 2} \
      {
       set fileCount [expr $qzSize*$zT*$qaSize*$aT]
      } \
   elseif {$dimCount == 4 && $fmtCount == 1} \
      {
       set fileCount [expr $qaSize*$aT]
       set cubeFlag 1
      } \
   elseif {$dimCount == 4 && $fmtCount == 0} \
      {
       set pipeFlag 1
      } \
   elseif {$dimCount == 3 && $fmtCount == 1} \
      {
       set fileCount [expr $qzSize*$zT]
      } \
   elseif {$dimCount == 3 && $fmtCount == 0} \
      {
       set pipeFlag 1
      }

   setParm $hdrBuff FDDIMCOUNT  $dimCount  NULL_DIM
   setParm $hdrBuff FDPIPEFLAG  $pipeFlag  NULL_DIM
   setParm $hdrBuff FDCUBEFLAG  $cubeFlag  NULL_DIM
   setParm $hdrBuff FDFILECOUNT $fileCount NULL_DIM

   if {$dimCount >= 2} \
      {
       setParm $hdrBuff NDSIZE   [expr $qySize*$yT] CUR_YDIM
       setParm $hdrBuff NDAPOD   $yT                CUR_YDIM

       if {$aq2DMode == 0 || ($qxSize == 1 && $qySize == 1)} \
          {
           setParm $hdrBuff NDQUADFLAG 1.0 CUR_YDIM
          } \
       else \
          {
           setParm $hdrBuff NDQUADFLAG 0.0 CUR_YDIM
          }

       updateOrig $hdrBuff $yT CUR_YDIM
      }

   if {$dimCount >= 3} \
      {
       setParm $hdrBuff NDSIZE   [expr $qzSize*$zT] CUR_ZDIM
       setParm $hdrBuff NDAPOD   $zT                CUR_ZDIM

       if {$qzSize == 1} \
          {
           setParm $hdrBuff NDQUADFLAG 1.0 CUR_ZDIM
          } \
       else \
          {
           setParm $hdrBuff NDQUADFLAG 0.0 CUR_ZDIM
          }

       updateOrig $hdrBuff $zT CUR_ZDIM
      }

   if {$dimCount >= 4} \
      {
       setParm $hdrBuff NDSIZE   [expr $qaSize*$aT] CUR_ADIM
       setParm $hdrBuff NDAPOD   $aT                CUR_ADIM

       if {$qaSize == 1} \
          {
           setParm $hdrBuff NDQUADFLAG 1.0 CUR_ADIM
          } \
       else \
          {
           setParm $hdrBuff NDQUADFLAG 0.0 CUR_ADIM
          }

       updateOrig $hdrBuff $aT CUR_ADIM
      }

   return 0
}

proc getInputHdr {} \
{
   global inUnit bytesPerWord inType inMode hdrSize padSize hdrBuff skipBuff FDATASIZE
   global acqus acqu2s a1Name a2Name inType

   if {![strcasecmp $inMode pipe]} \
      {
       set error    [rdFDATAU $inUnit hdrBuff]
       set inType   float
       set hdrSize  $FDATASIZE
       set padSize  0
      } \
   elseif {![strcasecmp $inMode varian]} \
      {
       if {$hdrSize} \
          {
           set hdrBuff [sysMalloc $hdrSize float]
           set status  [rdInput   $inUnit $hdrBuff $hdrSize]
           set vInfo   [getVHdrInfo $hdrBuff]

           if {[lsearch -exact $vInfo float] >= 0} \
              {
               if {![strcasecmp $inType Auto]} {set inType float}
              }

           if {![strcasecmp $inType Auto]} {set inType int}
          }
      } \
   elseif {![strcasecmp $inMode bruker]} \
      {
       if {[file exists $a1Name]} {rdJCAMP $a1Name acqus}
       if {[file exists $a2Name]} {rdJCAMP $a2Name acqu2s}

       if {[info exists acqus(DTYPA)]} \
          {
           if {[string is integer $acqus(DTYPA)]} \
              {
               if {$acqus(DTYPA) == 2} \
                  {
                   if {![strcasecmp $inType Auto]} {set inType double}
                  } \
               elseif {$acqus(DTYPA) == 1} \
                  {
                   if {![strcasecmp $inType Auto]} {set inType float}
                  }
              }
          }

      if {![strcasecmp $inType Auto]} {set inType int}

      if {$hdrSize} \
         {
          set hdrBuff [sysMalloc $hdrSize $inType]
          set status  [rdInput   $inUnit $hdrBuff $hdrSize]
         }
      } \
   else \
      {
       if {[strcasecmp $inType Auto]} {set inType int}

       if {$hdrSize} \
          {
           set hdrBuff [sysMalloc $hdrSize $inType]
           set status  [rdInput   $inUnit $hdrBuff $hdrSize]
          }
      }

   if {![strcasecmp $inType double]} {set bytesPerWord 8}

   if {$padSize} {set skipBuff [sysMalloc $padSize $inType]}
}

proc getInputData {} \
{
   global inName inUnit inType xSize qSize ySize inBuff sumBuff skipBuff padSize 
   global wBuff nBuff zeroBuff oneBuff nTotal nZero maskFlag verbose

   set inBuff    0
   set sumBuff   0
   set zeroBuff  0
   set oneBuff   0
   set wBuff     0
   set nBuff     0

   set nTotal    0
   set nZero     0

   if {![strcasecmp $xSize Auto] || ![strcasecmp $ySize Auto]} \
      {
       puts stderr "NUS Expand Error: Missing -xSize or -ySize"
       exit 1 
      }

   set vSize  [expr $qSize*$xSize]
   set nTotal [expr $vSize*$ySize]
   set nZero  [expr $xSize*16]

   if {$maskFlag} \
      {
       set nInput $vSize
      } \
   else \
      {
       set nInput $nTotal
      }

   set error [catch {set inBuff [sysMalloc $nInput $inType]}]

   if {$error} \
      {
       puts stderr "NUS Expand Error Allocating Input Buffer."
       exit 1
      }

   set error [catch {set sumBuff [sysMalloc $nZero $inType]}]

   if {$error} \
      {
       puts stderr "NUS Expand Error Allocating Sum Buffer."
       exit 1
      }

   set error [catch {set zeroBuff [sysMalloc $nZero $inType]}]

   if {$error} \
      {
       puts stderr "NUS Expand Error Allocating Zero Buffer."
       exit 1
      }

   set error [catch {set oneBuff [sysMalloc $nZero $inType]}]

   if {$error} \
      {
       puts stderr "NUS Expand Error Allocating Mask Buffer."
       exit 1
      }

   set error [catch {set nBuff [sysMalloc $nZero $inType]}]

   if {$error} \
      {
       puts stderr "NUS Expand Error Allocating vNeg Buffer."
       exit 1
      }

   set error [catch {set wBuff [sysMalloc $nZero float]}]

   if {$error} \
      {
       puts stderr "NUS Expand Error Allocating Weight Buffer."
       exit 1
      }

   for {set i 0} {$i < $nZero} {incr i} \
      {
       setPtrVal $zeroBuff $i 0 $inType
       setPtrVal $oneBuff  $i 1 $inType
      }

   set inPtr $inBuff

   if {$maskFlag} \
      {
       if {$verbose} \
          {
           puts stderr ""
           puts stderr "NUS Expand. Mask Mode, No Reading of Input Required."
          }
      } \
   else \
      {
       if {$verbose} \
          {
           puts stderr ""
           puts stderr "NUS Expand. Reading $nTotal Points ..."
          }

       if {$padSize} \
          {
           for {set i 0} {$i < $ySize} {incr i} \
              {
               set status [rdInput $inUnit $skipBuff $padSize]
               set status [rdInput $inUnit $inPtr    $vSize]
               set inPtr  [incrPtr $inPtr $vSize]
              }
          } \
       else \
          {
           set status [rdInput $inUnit $inBuff $nTotal]
          }
      }

   if {$inUnit} {sysClose $inUnit}
}

proc saveProf {} \
{
   global inMode profName hdrBuff fdataP FDATASIZE profBuff nProf xT yT zT aT dimCount verbose
   global yInfo zInfo aInfo wMin wMax sList wList profWFlag

   if {![strcasecmp $profName None]} {return 0}

   set nusDim [expr $dimCount - 1]

   if {![strcasecmp $profName Auto]} \
      {
       if {$dimCount == 2} \
          {
           set profName profY.dat
          } \
       elseif {$dimCount == 3} \
          {
           set profName profYZ.dat
          } \
       elseif {$dimCount == 4} \
          {
           set profName profYZA.dat
          } \
       else \
          {
           set profName None
           return
          }
      }

   set fmtCount [getFmtCount $profName]

   if {$fmtCount} \
      {
       set thisDir [file dirname $profName]
       if {![file isdirectory $thisDir]} \
          {
           if {$verbose} {puts stderr "NUS Expand. Creating Directory $thisDir ..."}
           catch "file mkdir $thisDir"
          }
      }

   if {$verbose} {puts stderr "NUS Expand. Creating ${nusDim}D Sampling Kernel $profName ..."}

   set error [catch {set profBuff [sysMalloc $yT        float]}]
   set error [catch {set fdataP   [sysMalloc $FDATASIZE float]}]

   set nProf $yT

   set thisName [format $profName 1]
   set dirName  [file dirname $thisName]

   if {![file exists $dirName]} {catch {file mkdir $dirName}}
   
   if {![file isdirectory $dirName]} \
      {
       puts stderr "NUS Expand Error Finding Sampling Schedule Output Directory $dirName"
       exit 1
      }

   set yLab Y_INC
   set zLab Z_INC
   set aLab A_INC

   set ySW  $yT
   set zSW  $zT
   set aSW  $aT

   set yObs 1.0
   set zObs 1.0
   set aObs 1.0

   set yCar 0.0
   set zCar 0.0
   set aCar 0.0

   if {![strcasecmp $inMode pipe]} \
      {
       set yLab [getParmStr $hdrBuff NDLABEL CUR_YDIM] 
       set zLab [getParmStr $hdrBuff NDLABEL CUR_ZDIM] 
       set aLab [getParmStr $hdrBuff NDLABEL CUR_ADIM] 

       set ySW  [getParm $hdrBuff NDSW  CUR_YDIM] 
       set zSW  [getParm $hdrBuff NDSW  CUR_ZDIM] 
       set aSW  [getParm $hdrBuff NDSW  CUR_ADIM] 

       set yObs [getParm $hdrBuff NDOBS CUR_YDIM] 
       set zObs [getParm $hdrBuff NDOBS CUR_ZDIM] 
       set aObs [getParm $hdrBuff NDOBS CUR_ADIM] 

       set yCar [getParm $hdrBuff NDCAR CUR_YDIM] 
       set zCar [getParm $hdrBuff NDCAR CUR_ZDIM] 
       set aCar [getParm $hdrBuff NDCAR CUR_ADIM] 
      }

   switch -exact $nusDim \
      {
       1 {
          buildHdr $fdataP -ndim $nusDim -aq2D Real \
             -xMODE Real -xFT Time -xN  $yT -xT $yT -xSW $ySW -xOBS $yObs -xCAR $yCar -xLAB $yLab 
         }

       2 {
          buildHdr $fdataP -ndim $nusDim -aq2D Real \
             -xMODE Real -xFT Time -xN  $yT -xT $yT -xSW $ySW -xOBS $yObs -xCAR $yCar -xLAB $yLab \
             -yMODE Real -yFT Time -yN  $zT -yT $zT -ySW $zSW -yOBS $zObs -yCAR $zCar -yLAB $zLab 
         }

       3 {
          buildHdr $fdataP -ndim $nusDim -aq2D Real \
             -xMODE Real -xFT Time -xN  $yT -xT $yT -xSW $ySW -xOBS $yObs -xCAR $yCar -xLAB $yLab \
             -yMODE Real -yFT Time -yN  $zT -yT $zT -ySW $zSW -yOBS $zObs -yCAR $zCar -yLAB $zLab \
             -zMODE Real -zFT Time -zN  $aT -zT $aT -zSW $aSW -zOBS $aObs -zCAR $aCar -zLAB $aLab  

          if {$fmtCount} \
             {
              setParm $fdataP FDFILECOUNT $aT NULL_DIM
             } \
          else \
             {
              setParm $fdataP FDPIPEFLAG  1.0 NULL_DIM
             }
         }
      }

   setParm $fdataP FDMIN       $wMin NULL_DIM
   setParm $fdataP FDMAX       $wMax NULL_DIM
   setParm $fdataP FDSCALEFLAG 1.0   NULL_DIM

   set nusDim [expr $dimCount - 1]

   for {set ia 0} {$ia < $aT} {incr ia} \
      {
       if {($ia == 0) || ($fmtCount != 0)} \
          {
           set thisName [format $profName [expr $ia + 1]]
           set error    [createPath $thisName]
           set error    [catch {set pUnit [sysOpen $thisName w]}]

           if {$error} \
              {
               puts stderr "NUS Expand Error Opening Sampling Kernel Output File $thisName"
               exit 1
              }

           wrOutput $pUnit $fdataP $FDATASIZE
          }

       for {set iz 0} {$iz < $zT} {incr iz} \
          {
           vsSet $profBuff 0.0 $nProf

           foreach iy $yInfo($ia,$iz) \
             {
              set thisW 1.0

              if {$profWFlag} \
                 {
                  set iList  [join [lrange "$iy $iz $ia" 0 [expr $nusDim - 1]] ","]
                  set loc    [lsearch -exact $sList $iList]

                  if {$loc >= 0} {set thisW [lindex $wList $loc]}
                 }

              setPtrVal $profBuff $iy $thisW 
             }

           wrOutput $pUnit $profBuff $nProf
          }

       if {($ia == $aT - 1) || ($fmtCount != 0)} {sysClose $pUnit}
      }
}

proc initOutputPlane { iz ia } \
{
   global outName currentOutName outUnit dimCount inMode hdrBuff hdrSize FDATASIZE verbose

   if {![strcasecmp $outName None]} return

   if {![strcasecmp $outName Auto]} {set outName nus_full}

   if {![strcasecmp $inMode pipe]} \
      {
       set wFlag 0
       set oFlag 0

       if {![strcasecmp $outName stdout]} \
          {
           if {$iz == 1 && $ia == 1} {set wFlag 1}
           set currentOutName $outName
          } \
       else \
          {
           set fmtCount [getFmtCount $outName]

           if {$dimCount == 4 && $fmtCount == 2} \
              {
               set thisName [format $outName $ia $iz]
               set oFlag    1
              } \
           elseif {$dimCount == 4 && $fmtCount == 1} \
              {
               set thisName [format $outName $ia]
               if {$iz == 1} {set oFlag 1}
              } \
           elseif {$dimCount == 4 && $fmtCount == 0} \
              {
               set thisName $outName
               if {$iz == 1 && $ia == 1} {set oFlag 1}
              } \
           elseif {$dimCount == 3 && $fmtCount == 1} \
              {
               set thisName [format $outName $iz]
               set oFlag    1
              } \
           else \
              {
               set thisName $outName
               if {$iz == 1 && $ia == 1} {set oFlag 1}
              }

           if {$oFlag} \
              {
               set wFlag 1

               if {[strcmp $outUnit 1] && [strcmp $outUnit None]} \
                  {
                   catch "sysClose $outUnit"
                   set outUnit None
                  }

               if {[file exists $thisName]} {catch "file delete $thisName"}

               set error [createPath $thisName] 

#              if {$verbose} {puts -nonewline stderr "NUS Expand. Opening Output:       $thisName\r"}

               set error [catch {set outUnit [sysOpen $thisName w]}]

               if {$error} \
                  {
                   puts stderr "NUS Expand Error Opening Output File $thisName"
                   exit 1
                  }

               set currentOutName $thisName
              }
           }

       if {$wFlag} \
          {
           wrOutput $outUnit $hdrBuff $FDATASIZE
          }
      } \
   else \
      {
       if {$iz != 1 || $ia != 1} return

       if {[strcasecmp $outName stdout]} \
          {
           if {[strcmp $outUnit 1] && [strcasecmp $outUnit None]} \
              {
               catch "sysClose $outUnit"
               set outUnit 1
              }

           set thisName $outName
           set error [createPath $thisName] 

           if {[file exists $thisName]} {catch "file delete $thisName"}

           if {$verbose} {puts stderr "NUS Expand. Opening Output File: $thisName"}

           set error [catch {set outUnit [sysOpen $thisName w]}]

           if {$error} \
              {
               puts stderr "NUS Expand Error Opening Output File $outName"
               exit 1
              }
          }

       if {![strcasecmp $inMode varian]} {wrOutput $outUnit $hdrBuff $hdrSize}
      }
}

proc getNUSSign { iyq izq iaq } \
{
   global maskFlag signFlag defSign aqORD dimCount

   set thisSign $defSign 

   if {!$signFlag}     {return $thisSign}
   if {$maskFlag}      {return $thisSign}
   if {$dimCount == 2} {return $thisSign}

   if {$aqORD == 1} \
      {
       if {$iyq == 1} {set thisSign [expr -$thisSign]} 
# %%%
       if {$izq == 1} {set thisSign [expr -$thisSign]} 
       if {$izq == 1 && $dimCount == 4} {set thisSign [expr -$thisSign]} 
      } \
   else \
      {
       if {$izq == 1} {set thisSign [expr -$thisSign]} 
       if {$iaq == 1} {set thisSign [expr -$thisSign]} 
      }

   return $thisSign
}

proc saveNUSVec { ia iaq iz izq iy iyq } \
{
   global inBuff inType sumBuff zeroBuff oneBuff nBuff wBuff skipBuff nZero padSize vSize outName outUnit verbose
   global sList wList nusDim dimCount minLocCount maxLocCount maxOff maxUsed maskFlag multWFlag avgFlag nBad

   set iList    [join [lrange "$iy $iz $ia" 0 [expr $nusDim - 1]] ","]
   set locList  [lsearch -exact -all $sList $iList]
   set locCount [llength $locList]

   if {$minLocCount < 0} \
      {
       set minLocCount $locCount
       set maxLocCount $locCount
      }

   if {$locCount < $minLocCount} {set minLocCount $locCount}
   if {$locCount > $maxLocCount} {set maxLocCount $locCount}

   set src       $zeroBuff
   set srcStr    zero
   set thisW     0.0
   set off       0
   set usedCount 0

   if {$locCount > 1} {vsSet $sumBuff 0.0 $nZero}

   foreach loc $locList \
      {
       set off   [getNUSOffset $ia $iaq $iz $izq $iy $iyq $loc] 
       set thisW [lindex $wList $loc]

       if {$off > $maxOff} \
          {
           puts stderr ""
           puts stderr "NUS Expand Warning: Bad Sample:" 
           puts stderr " Y: $iy $iyq Z: $iz $izq A: $ia $iaq Sample: $loc Src Offset: $srcStr"
           puts stderr ""

           set srcStr error 
           incr nBad
          } \
       else \
          {
           if {$thisW == 0.0} \
              {
               set src $zeroBuff
               set srcStr zero 
              } \
           elseif {$maskFlag} \
              {
               set src $oneBuff
               set srcStr one
               incr usedCount
              } \
           else \
              {
               set src [incrPtr $inBuff $off]
               set srcStr $off
               incr usedCount
              }

           if {$multWFlag} \
              {
               if {![strcasecmp $inType float]} \
                  {
                   vsMult $wBuff $src $thisW $vSize
                  }  \
               elseif {![strcasecmp $inType int]} \
                  {
                   vsMultInt $wBuff $src $thisW $vSize
                  } \
               elseif {![strcasecmp $inType double]} \
                  {
                   vsMultDouble $wBuff $src $thisW $vSize
                  } \
               else \
                  {
                   puts stderr ""
                   puts stderr "NUS Expand Error: Weighting Option Requires Input Type float int or double."
                   exit 0
                  }

               set src $wBuff
              } 
          }

       if {$avgFlag && $maskFlag} \
          {
           set src $src
          } \
       elseif {$locCount > 1} \
          {
           if {![strcasecmp $inType float]} \
              {
               vvAdd $sumBuff $src $vSize
              } \
           elseif {![strcasecmp $inType int]} \
              {
               vvAddInt $sumBuff $src $vSize
              } \
           elseif {![strcasecmp $inType double]} \
              {
               vvAddDouble $sumBuff $src $vSize
              } \
           else \
              {
               puts stderr ""
               puts stderr "NUS Expand Error: Non-Uniform Averaging Requires Input Type float int or double."
               exit 0
              }

           set src $sumBuff
          }
      }

   if {$avgFlag && !$maskFlag && $usedCount > 1} \
      {
       if {![strcasecmp $inType float]} \
          {
           vsMult $src [expr 1.0/$usedCount] $vSize
          }  \
       elseif {![strcasecmp $inType int]} \
          {
           vsMultInt $src [expr 1.0/$usedCount] $vSize
          } \
       elseif {![strcasecmp $inType double]} \
          {
           vsMultDouble $src [expr 1.0/$usedCount] $vSize
          } \
       else \
          {
           puts stderr ""
           puts stderr "NUS Expand Error: -avg Option Requires Input Type float int or double."
           exit 0
          }
      }

   set thisSign [getNUSSign $iyq $izq $iaq]
   
   if {$thisSign < 0} \
      {
       if {![strcasecmp $inType float]} \
          {
           vNeg $nBuff $src $vSize
          } \
       elseif {![strcasecmp $inType int]} \
          {
           vNegInt $nBuff $src $vSize
          } \
       elseif {![strcasecmp $inType double]} \
          {
           vNegDouble $nBuff $src $vSize
          } \
       else \
          {
           puts stderr ""
           puts stderr "NUS Expand Error: Sign Adjustment Requires Input Type float int or double."
           exit 0
          }

       set src $nBuff
      }

   if {$off > $maxUsed} {set maxUsed $off}

   if {[strcasecmp $outName None]} \
      {
       wrOutput $outUnit $skipBuff $padSize
       wrOutput $outUnit $src      $vSize
      }

   return $locCount
}

#
#------------------------------------------------------------------------------

set sList     "" 
set wList     "" 

set fdataP      0
set inBuff      0
set sumBuff     0
set zeroBuff    0
set profBuff    0
set hdrBuff     0
set skipBuff    0
set wMin        0.0
set wMax        1.0

set maxOff       0
set maxLocCount -1 
set minLocCount -1 

set aq2DMode  -1

set nTotal    0
set nZero     0
set nProf     0

set inUnit    0
set outUnit   1

if {$verbose} \
   {
    puts stderr ""
    puts stderr "NUS Expand. Version:              $thisVersion"
    puts stderr "NUS Expand. Binary Type:          [getNMRBinType]"
   }

initInput
getInputHdr
getSampleParm 
getRPDSchedule
getSampleSchedule
setLocInfo
getInputData
setOutputHdr
saveProf

#
# Write the data in sorted, expanded order:

set nusDim     [expr $dimCount - 1]
set vSize      [expr $xSize*$qxSize]

if {$expandFlag} \
   {
    set maxOff [expr $nTotal - $vSize]
   } \
else \
   {
    set maxOff [expr $nTotal - $vSize]
   }

set maxUsed 0
set off     0

set nBad    0
set iip     0
set iia     0
set iiz     0

set totalLocs    0
set totalSampled 0
 
if {$verbose} \
   {
    puts stderr "NUS Expand. Sizes V: $vSize X: $xT $qxSize Y: $yT $qySize Z: $zT $qzSize A: $aT $qaSize"
    puts stderr ""
   }

if {![strcasecmp $inMode varian]} \
   {
    set planeCount [expr $zT*$aT]

    for {set ia 0} {$ia < $aT} {incr ia} \
     {
      set iiz 0
      incr iia

      for {set iz 0} {$iz < $zT} {incr iz} \
       {
        incr iiz
        incr iip

        initOutputPlane $iiz $iia

        if {$verbose} {puts -nonewline stderr "Plane $iip of $planeCount $currentOutName\015"}

        for {set iy 0} {$iy < $yT} {incr iy} \
         {
          for {set iaq 0} {$iaq < $qaSize} {incr iaq} \
           {
            for {set izq 0} {$izq < $qzSize} {incr izq} \
             {
              for {set iyq 0} {$iyq < $qySize} {incr iyq} \
               {
                set locCount [saveNUSVec $ia $iaq $iz $izq $iy $iyq]

                if {$locCount} {incr totalSampled}
                incr totalLocs
               }
             }
           }
         }
       }
     }
   } \
else \
   {
    set planeCount [expr $zT*$qzSize*$aT*$qaSize]

    for {set ia 0} {$ia < $aT} {incr ia} \
     {
      for {set iaq 0} {$iaq < $qaSize} {incr iaq} \
       {
        set iiz 0
        incr iia

        for {set iz 0} {$iz < $zT} {incr iz} \
         {
          for {set izq 0} {$izq < $qzSize} {incr izq} \
          {
           incr iiz
           incr iip 

           initOutputPlane $iiz $iia

           if {$verbose} {puts -nonewline stderr "Plane $iip of $planeCount $currentOutName\015"}

           for {set iy 0} {$iy < $yT} {incr iy} \
            {
             for {set iyq 0} {$iyq < $qySize} {incr iyq} \
             {
              set  locCount [saveNUSVec $ia $iaq $iz $izq $iy $iyq]

              if {$locCount} {incr totalSampled}
              incr totalLocs
             }
            }
          }
         }
       }
     }
   }

if {$verbose} \
   {
    puts stderr ""
    puts stderr ""
    puts stderr "NUS Expand. Output Min Samples: $minLocCount"
    puts stderr "NUS Expand. Output Max Samples: $maxLocCount"
    puts stderr "NUS Expand. Max Allowed Offset: $maxOff Used: $maxUsed"

    if {$maxLocCount > 1} \
       {
        puts stderr ""

        if {$avgFlag} \
           {
            puts stderr "NUS Expand. Notice: Redundant Increments Detected."
            puts stderr "NUS Expand. Notice: Redundant Increments Have Been Averaged."
           } \
        else \
           {
            puts stderr "NUS Expand. Warning: Redundant Increments Detected."
            puts stderr "NUS Expand. Warning: Redundant Increments Have Been Summed."
           }
       }

    puts stderr ""
   }

if {$nBad} {puts stderr "NUS Expand Warning: $nBad Bad Sample Locations."}

if {[strcmp $outUnit 1] && [strcmp $outUnit None]} {catch "sysClose $outUnit"}

exit 0

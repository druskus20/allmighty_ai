Øō:
ż
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.1.02unknown8żĒ6
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	<*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:<*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<(*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:<(*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:(*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:(*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0

time_distributed/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nametime_distributed/kernel

+time_distributed/kernel/Read/ReadVariableOpReadVariableOptime_distributed/kernel*&
_output_shapes
: *
dtype0

time_distributed/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametime_distributed/bias
{
)time_distributed/bias/Read/ReadVariableOpReadVariableOptime_distributed/bias*
_output_shapes
: *
dtype0

time_distributed_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nametime_distributed_2/kernel

-time_distributed_2/kernel/Read/ReadVariableOpReadVariableOptime_distributed_2/kernel*&
_output_shapes
: *
dtype0

time_distributed_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nametime_distributed_2/bias

+time_distributed_2/bias/Read/ReadVariableOpReadVariableOptime_distributed_2/bias*
_output_shapes
:*
dtype0
r

gru/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
	*
shared_name
gru/kernel
k
gru/kernel/Read/ReadVariableOpReadVariableOp
gru/kernel* 
_output_shapes
:
	*
dtype0

gru/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_namegru/recurrent_kernel

(gru/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/recurrent_kernel* 
_output_shapes
:
*
dtype0
m
gru/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_name
gru/bias
f
gru/bias/Read/ReadVariableOpReadVariableOpgru/bias*
_output_shapes
:	*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

RMSprop/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<*)
shared_nameRMSprop/dense/kernel/rms

,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms*
_output_shapes
:	<*
dtype0

RMSprop/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameRMSprop/dense/bias/rms
}
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
_output_shapes
:<*
dtype0

RMSprop/dense_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<(*+
shared_nameRMSprop/dense_1/kernel/rms

.RMSprop/dense_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/kernel/rms*
_output_shapes

:<(*
dtype0

RMSprop/dense_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*)
shared_nameRMSprop/dense_1/bias/rms

,RMSprop/dense_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/bias/rms*
_output_shapes
:(*
dtype0

RMSprop/dense_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*+
shared_nameRMSprop/dense_2/kernel/rms

.RMSprop/dense_2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/kernel/rms*
_output_shapes

:(*
dtype0

RMSprop/dense_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_2/bias/rms

,RMSprop/dense_2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/bias/rms*
_output_shapes
:*
dtype0
Ŗ
#RMSprop/time_distributed/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#RMSprop/time_distributed/kernel/rms
£
7RMSprop/time_distributed/kernel/rms/Read/ReadVariableOpReadVariableOp#RMSprop/time_distributed/kernel/rms*&
_output_shapes
: *
dtype0

!RMSprop/time_distributed/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!RMSprop/time_distributed/bias/rms

5RMSprop/time_distributed/bias/rms/Read/ReadVariableOpReadVariableOp!RMSprop/time_distributed/bias/rms*
_output_shapes
: *
dtype0
®
%RMSprop/time_distributed_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%RMSprop/time_distributed_2/kernel/rms
§
9RMSprop/time_distributed_2/kernel/rms/Read/ReadVariableOpReadVariableOp%RMSprop/time_distributed_2/kernel/rms*&
_output_shapes
: *
dtype0

#RMSprop/time_distributed_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#RMSprop/time_distributed_2/bias/rms

7RMSprop/time_distributed_2/bias/rms/Read/ReadVariableOpReadVariableOp#RMSprop/time_distributed_2/bias/rms*
_output_shapes
:*
dtype0

RMSprop/gru/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
	*'
shared_nameRMSprop/gru/kernel/rms

*RMSprop/gru/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/gru/kernel/rms* 
_output_shapes
:
	*
dtype0

 RMSprop/gru/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" RMSprop/gru/recurrent_kernel/rms

4RMSprop/gru/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp RMSprop/gru/recurrent_kernel/rms* 
_output_shapes
:
*
dtype0

RMSprop/gru/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameRMSprop/gru/bias/rms
~
(RMSprop/gru/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/gru/bias/rms*
_output_shapes
:	*
dtype0

NoOpNoOp
üP
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*·P
value­PBŖP B£P
«
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
m
	layer

_input_map
regularization_losses
trainable_variables
	variables
	keras_api
m
	layer

_input_map
regularization_losses
trainable_variables
	variables
	keras_api
m
	layer
 
_input_map
!regularization_losses
"trainable_variables
#	variables
$	keras_api
m
	%layer
&
_input_map
'regularization_losses
(trainable_variables
)	variables
*	keras_api
m
	+layer
,
_input_map
-regularization_losses
.trainable_variables
/	variables
0	keras_api
m
	1layer
2
_input_map
3regularization_losses
4trainable_variables
5	variables
6	keras_api
l
7cell
8
state_spec
9regularization_losses
:trainable_variables
;	variables
<	keras_api
R
=regularization_losses
>trainable_variables
?	variables
@	keras_api
h

Akernel
Bbias
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
h

Gkernel
Hbias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
h

Mkernel
Nbias
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
Ū
Siter
	Tdecay
Ulearning_rate
Vmomentum
Wrho
ArmsÓ
BrmsŌ
GrmsÕ
HrmsÖ
Mrms×
NrmsŲ
XrmsŁ
YrmsŚ
ZrmsŪ
[rmsÜ
\rmsŻ
]rmsŽ
^rmsß
 
^
X0
Y1
Z2
[3
\4
]5
^6
A7
B8
G9
H10
M11
N12
^
X0
Y1
Z2
[3
\4
]5
^6
A7
B8
G9
H10
M11
N12

regularization_losses

_layers
trainable_variables
`layer_regularization_losses
anon_trainable_variables
bmetrics
	variables
 
h

Xkernel
Ybias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
 
 

X0
Y1

X0
Y1

regularization_losses
trainable_variables
glayer_regularization_losses
hnon_trainable_variables
imetrics
	variables

jlayers
R
kregularization_losses
ltrainable_variables
m	variables
n	keras_api
 
 
 
 

regularization_losses
trainable_variables
olayer_regularization_losses
pnon_trainable_variables
qmetrics
	variables

rlayers
h

Zkernel
[bias
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
 
 

Z0
[1

Z0
[1

!regularization_losses
"trainable_variables
wlayer_regularization_losses
xnon_trainable_variables
ymetrics
#	variables

zlayers
R
{regularization_losses
|trainable_variables
}	variables
~	keras_api
 
 
 
 

'regularization_losses
(trainable_variables
layer_regularization_losses
non_trainable_variables
metrics
)	variables
layers
V
regularization_losses
trainable_variables
	variables
	keras_api
 
 
 
 

-regularization_losses
.trainable_variables
 layer_regularization_losses
non_trainable_variables
metrics
/	variables
layers
V
regularization_losses
trainable_variables
	variables
	keras_api
 
 
 
 

3regularization_losses
4trainable_variables
 layer_regularization_losses
non_trainable_variables
metrics
5	variables
layers


\kernel
]recurrent_kernel
^bias
regularization_losses
trainable_variables
	variables
	keras_api
 
 

\0
]1
^2

\0
]1
^2

9regularization_losses
:trainable_variables
 layer_regularization_losses
non_trainable_variables
metrics
;	variables
layers
 
 
 

=regularization_losses
>trainable_variables
 layer_regularization_losses
non_trainable_variables
metrics
?	variables
layers
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1

A0
B1

Cregularization_losses
Dtrainable_variables
 layer_regularization_losses
 non_trainable_variables
”metrics
E	variables
¢layers
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

G0
H1

Iregularization_losses
Jtrainable_variables
 £layer_regularization_losses
¤non_trainable_variables
„metrics
K	variables
¦layers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

M0
N1

M0
N1

Oregularization_losses
Ptrainable_variables
 §layer_regularization_losses
Ønon_trainable_variables
©metrics
Q	variables
Ŗlayers
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEtime_distributed/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEtime_distributed/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEtime_distributed_2/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEtime_distributed_2/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
gru/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEgru/recurrent_kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEgru/bias0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
N
0
1
2
3
4
5
6
	7

8
9
10
 
 

«0
 

X0
Y1

X0
Y1

cregularization_losses
dtrainable_variables
 ¬layer_regularization_losses
­non_trainable_variables
®metrics
e	variables
Ælayers
 
 
 

0
 
 
 

kregularization_losses
ltrainable_variables
 °layer_regularization_losses
±non_trainable_variables
²metrics
m	variables
³layers
 
 
 

0
 

Z0
[1

Z0
[1

sregularization_losses
ttrainable_variables
 “layer_regularization_losses
µnon_trainable_variables
¶metrics
u	variables
·layers
 
 
 

0
 
 
 

{regularization_losses
|trainable_variables
 ølayer_regularization_losses
¹non_trainable_variables
ŗmetrics
}	variables
»layers
 
 
 

%0
 
 
 
”
regularization_losses
trainable_variables
 ¼layer_regularization_losses
½non_trainable_variables
¾metrics
	variables
ælayers
 
 
 

+0
 
 
 
”
regularization_losses
trainable_variables
 Ąlayer_regularization_losses
Įnon_trainable_variables
Āmetrics
	variables
Ćlayers
 
 
 

10
 

\0
]1
^2

\0
]1
^2
”
regularization_losses
trainable_variables
 Älayer_regularization_losses
Ånon_trainable_variables
Ęmetrics
	variables
Ēlayers
 
 
 

70
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


Čtotal

Écount
Ź
_fn_kwargs
Ėregularization_losses
Ģtrainable_variables
Ķ	variables
Ī	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

Č0
É1
”
Ėregularization_losses
Ģtrainable_variables
 Ļlayer_regularization_losses
Šnon_trainable_variables
Ńmetrics
Ķ	variables
Ņlayers
 

Č0
É1
 
 

VARIABLE_VALUERMSprop/dense/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/dense/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/dense_1/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUERMSprop/dense_1/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/dense_2/kernel/rmsTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUERMSprop/dense_2/bias/rmsRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#RMSprop/time_distributed/kernel/rmsNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!RMSprop/time_distributed/bias/rmsNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%RMSprop/time_distributed_2/kernel/rmsNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#RMSprop/time_distributed_2/bias/rmsNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUERMSprop/gru/kernel/rmsNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE RMSprop/gru/recurrent_kernel/rmsNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUERMSprop/gru/bias/rmsNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*5
_output_shapes#
!:’’’’’’’’’ī*
dtype0**
shape!:’’’’’’’’’ī

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1time_distributed/kerneltime_distributed/biastime_distributed_2/kerneltime_distributed_2/bias
gru/kernelgru/recurrent_kernelgru/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference_signature_wrapper_33134
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp+time_distributed/kernel/Read/ReadVariableOp)time_distributed/bias/Read/ReadVariableOp-time_distributed_2/kernel/Read/ReadVariableOp+time_distributed_2/bias/Read/ReadVariableOpgru/kernel/Read/ReadVariableOp(gru/recurrent_kernel/Read/ReadVariableOpgru/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOp.RMSprop/dense_1/kernel/rms/Read/ReadVariableOp,RMSprop/dense_1/bias/rms/Read/ReadVariableOp.RMSprop/dense_2/kernel/rms/Read/ReadVariableOp,RMSprop/dense_2/bias/rms/Read/ReadVariableOp7RMSprop/time_distributed/kernel/rms/Read/ReadVariableOp5RMSprop/time_distributed/bias/rms/Read/ReadVariableOp9RMSprop/time_distributed_2/kernel/rms/Read/ReadVariableOp7RMSprop/time_distributed_2/bias/rms/Read/ReadVariableOp*RMSprop/gru/kernel/rms/Read/ReadVariableOp4RMSprop/gru/recurrent_kernel/rms/Read/ReadVariableOp(RMSprop/gru/bias/rms/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference__traced_save_36486
ģ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhotime_distributed/kerneltime_distributed/biastime_distributed_2/kerneltime_distributed_2/bias
gru/kernelgru/recurrent_kernelgru/biastotalcountRMSprop/dense/kernel/rmsRMSprop/dense/bias/rmsRMSprop/dense_1/kernel/rmsRMSprop/dense_1/bias/rmsRMSprop/dense_2/kernel/rmsRMSprop/dense_2/bias/rms#RMSprop/time_distributed/kernel/rms!RMSprop/time_distributed/bias/rms%RMSprop/time_distributed_2/kernel/rms#RMSprop/time_distributed_2/bias/rmsRMSprop/gru/kernel/rms RMSprop/gru/recurrent_kernel/rmsRMSprop/gru/bias/rms*-
Tin&
$2"*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__traced_restore_36597Ź5
ė
i
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34256

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ė         2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2	
Reshapeø
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:’’’’’’’’’/3 *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :/2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :32
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape¢
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’/3 2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’/3 2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:(’’’’’’’’’’’’’’’’’’ė :& "
 
_user_specified_nameinputs
±4

E__inference_sequential_layer_call_and_return_conditional_losses_33091

inputs3
/time_distributed_statefulpartitionedcall_args_13
/time_distributed_statefulpartitionedcall_args_25
1time_distributed_2_statefulpartitionedcall_args_15
1time_distributed_2_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_1&
"gru_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢(time_distributed/StatefulPartitionedCall¢*time_distributed_2/StatefulPartitionedCallā
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputs/time_distributed_statefulpartitionedcall_args_1/time_distributed_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*5
_output_shapes#
!:’’’’’’’’’ė *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_318842*
(time_distributed/StatefulPartitionedCall
"time_distributed_1/PartitionedCallPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’/3 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_319192$
"time_distributed_1/PartitionedCall
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_1/PartitionedCall:output:01time_distributed_2_statefulpartitionedcall_args_11time_distributed_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’,0*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_319602,
*time_distributed_2/StatefulPartitionedCall
"time_distributed_3/PartitionedCallPartitionedCall3time_distributed_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_319952$
"time_distributed_3/PartitionedCall
"time_distributed_4/PartitionedCallPartitionedCall+time_distributed_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_320392$
"time_distributed_4/PartitionedCall
"time_distributed_5/PartitionedCallPartitionedCall+time_distributed_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_320702$
"time_distributed_5/PartitionedCallŽ
gru/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_5/PartitionedCall:output:0"gru_statefulpartitionedcall_args_1"gru_statefulpartitionedcall_args_2"gru_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_328592
gru/StatefulPartitionedCallā
dropout_1/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_329052
dropout_1/PartitionedCall¹
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’<*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_329282
dense/StatefulPartitionedCallĒ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’(*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_329502!
dense_1/StatefulPartitionedCallÉ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_329732!
dense_2/StatefulPartitionedCallÖ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:’’’’’’’’’ī:::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ø
i
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_31910

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ė         2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2	
Reshapeø
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:’’’’’’’’’/3 *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   /   3       2
Reshape_1/shape
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’/3 2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:’’’’’’’’’/3 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’ė :& "
 
_user_specified_nameinputs
@
ą
'__forward_cudnn_gru_with_fallback_32467

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *<
api_implements*(gru_398d159f-3a18-4e10-83f6-a767482eb326*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_cudnn_gru_with_fallback_32332_324682
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias


!__inference__traced_restore_36597
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias#
assignvariableop_6_rmsprop_iter$
 assignvariableop_7_rmsprop_decay,
(assignvariableop_8_rmsprop_learning_rate'
#assignvariableop_9_rmsprop_momentum#
assignvariableop_10_rmsprop_rho/
+assignvariableop_11_time_distributed_kernel-
)assignvariableop_12_time_distributed_bias1
-assignvariableop_13_time_distributed_2_kernel/
+assignvariableop_14_time_distributed_2_bias"
assignvariableop_15_gru_kernel,
(assignvariableop_16_gru_recurrent_kernel 
assignvariableop_17_gru_bias
assignvariableop_18_total
assignvariableop_19_count0
,assignvariableop_20_rmsprop_dense_kernel_rms.
*assignvariableop_21_rmsprop_dense_bias_rms2
.assignvariableop_22_rmsprop_dense_1_kernel_rms0
,assignvariableop_23_rmsprop_dense_1_bias_rms2
.assignvariableop_24_rmsprop_dense_2_kernel_rms0
,assignvariableop_25_rmsprop_dense_2_bias_rms;
7assignvariableop_26_rmsprop_time_distributed_kernel_rms9
5assignvariableop_27_rmsprop_time_distributed_bias_rms=
9assignvariableop_28_rmsprop_time_distributed_2_kernel_rms;
7assignvariableop_29_rmsprop_time_distributed_2_bias_rms.
*assignvariableop_30_rmsprop_gru_kernel_rms8
4assignvariableop_31_rmsprop_gru_recurrent_kernel_rms,
(assignvariableop_32_rmsprop_gru_bias_rms
identity_34¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1Æ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*»
value±B®!B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesŠ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÓ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:::::::::::::::::::::::::::::::::*/
dtypes%
#2!	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_iterIdentity_6:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOp assignvariableop_7_rmsprop_decayIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp(assignvariableop_8_rmsprop_learning_rateIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOp#assignvariableop_9_rmsprop_momentumIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_rmsprop_rhoIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11¤
AssignVariableOp_11AssignVariableOp+assignvariableop_11_time_distributed_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12¢
AssignVariableOp_12AssignVariableOp)assignvariableop_12_time_distributed_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13¦
AssignVariableOp_13AssignVariableOp-assignvariableop_13_time_distributed_2_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14¤
AssignVariableOp_14AssignVariableOp+assignvariableop_14_time_distributed_2_biasIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOpassignvariableop_15_gru_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16”
AssignVariableOp_16AssignVariableOp(assignvariableop_16_gru_recurrent_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17
AssignVariableOp_17AssignVariableOpassignvariableop_17_gru_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20„
AssignVariableOp_20AssignVariableOp,assignvariableop_20_rmsprop_dense_kernel_rmsIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21£
AssignVariableOp_21AssignVariableOp*assignvariableop_21_rmsprop_dense_bias_rmsIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22§
AssignVariableOp_22AssignVariableOp.assignvariableop_22_rmsprop_dense_1_kernel_rmsIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23„
AssignVariableOp_23AssignVariableOp,assignvariableop_23_rmsprop_dense_1_bias_rmsIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24§
AssignVariableOp_24AssignVariableOp.assignvariableop_24_rmsprop_dense_2_kernel_rmsIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25„
AssignVariableOp_25AssignVariableOp,assignvariableop_25_rmsprop_dense_2_bias_rmsIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26°
AssignVariableOp_26AssignVariableOp7assignvariableop_26_rmsprop_time_distributed_kernel_rmsIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27®
AssignVariableOp_27AssignVariableOp5assignvariableop_27_rmsprop_time_distributed_bias_rmsIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28²
AssignVariableOp_28AssignVariableOp9assignvariableop_28_rmsprop_time_distributed_2_kernel_rmsIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29°
AssignVariableOp_29AssignVariableOp7assignvariableop_29_rmsprop_time_distributed_2_bias_rmsIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30£
AssignVariableOp_30AssignVariableOp*assignvariableop_30_rmsprop_gru_kernel_rmsIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31­
AssignVariableOp_31AssignVariableOp4assignvariableop_31_rmsprop_gru_recurrent_kernel_rmsIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32”
AssignVariableOp_32AssignVariableOp(assignvariableop_32_rmsprop_gru_bias_rmsIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32Ø
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp“
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33Į
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*
_input_shapes
: :::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
©
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_32900

inputs
identitya
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/random_uniform/maxµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformŖ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/subĮ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/random_uniform/mulÆ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv¢
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/mul
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:& "
 
_user_specified_nameinputs


while_cond_33263
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice-
)while_cond_33263___redundant_placeholder0-
)while_cond_33263___redundant_placeholder1-
)while_cond_33263___redundant_placeholder2-
)while_cond_33263___redundant_placeholder3-
)while_cond_33263___redundant_placeholder4
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2: : : : :’’’’’’’’’: :::::
”
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_36255

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:’’’’’’’’’:& "
 
_user_specified_nameinputs
ų
ć
>__inference_gru_layer_call_and_return_conditional_losses_31448

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
zerosŠ
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*_
_output_shapesM
K:’’’’’’’’’:’’’’’’’’’’’’’’’’’’:’’’’’’’’’: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference_standard_gru_312242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:’’’’’’’’’’’’’’’’’’	:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
¦
l
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_34488

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
Reshapeq
dropout/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/dropout/raten
dropout/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape
"dropout/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dropout/dropout/random_uniform/min
"dropout/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"dropout/dropout/random_uniform/maxŌ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	*
dtype02.
,dropout/dropout/random_uniform/RandomUniformŹ
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"dropout/dropout/random_uniform/subč
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:’’’’’’’’’	2$
"dropout/dropout/random_uniform/mulÖ
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2 
dropout/dropout/random_uniforms
dropout/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/dropout/sub/x
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/dropout/sub{
dropout/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/dropout/truediv/x
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/dropout/truedivÉ
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/dropout/GreaterEqual
dropout/dropout/mulMulReshape:output:0dropout/dropout/truediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/dropout/mul
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’	2
dropout/dropout/Cast¢
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/dropout/mul_1
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2
Reshape_1/shape
	Reshape_1Reshapedropout/dropout/mul_1:z:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
Ė
i
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_32060

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
flatten/Const
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2
flatten/Reshapew
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’     2
Reshape_1/shape
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2
	Reshape_1k
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
ń
Ø
'__inference_dense_2_layer_call_fn_36317

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_329732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’(::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ž3

)__inference_cudnn_gru_with_fallback_35281

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis·
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes *<
api_implements*(gru_65b137f1-510e-4453-a02f-9d4932b073b5*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias

N
2__inference_time_distributed_3_layer_call_fn_34464

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_319952
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’,0:& "
 
_user_specified_nameinputs

N
2__inference_time_distributed_5_layer_call_fn_34607

inputs
identityĘ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_302532
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’	:& "
 
_user_specified_nameinputs
Ž3

)__inference_cudnn_gru_with_fallback_29482

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis·
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes *<
api_implements*(gru_4417fa00-1c49-4e6a-ba14-dc50117fc1d4*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Ž3

)__inference_cudnn_gru_with_fallback_32720

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis·
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes *<
api_implements*(gru_e83772c8-2ae3-4ece-9026-dedf2427102b*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias


while_cond_32541
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice-
)while_cond_32541___redundant_placeholder0-
)while_cond_32541___redundant_placeholder1-
)while_cond_32541___redundant_placeholder2-
)while_cond_32541___redundant_placeholder3-
)while_cond_32541___redundant_placeholder4
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2: : : : :’’’’’’’’’: :::::
ę
ć
>__inference_gru_layer_call_and_return_conditional_losses_32470

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
zerosĒ
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*V
_output_shapesD
B:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference_standard_gru_322462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’	:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
K
š
__inference_standard_gru_34807

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢whilef
ReadVariableOpReadVariableOpbias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ż
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’	*
shrink_axis_mask2
strided_slice_1w
MatMul/ReadVariableOpReadVariableOpkernel* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*F
_output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
bodyR
while_body_34714*
condR
while_cond_34713*E
output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeņ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permÆ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimeø
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

IdentityĄ

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2

Identity_1²

Identity_2Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_2¢

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*T
_input_shapesC
A:’’’’’’’’’’’’’’’’’’	:’’’’’’’’’:::*<
api_implements*(gru_214693d3-9c8e-4a5e-a5f4-04ba364da9ac*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Ė
i
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_34632

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
flatten/Const
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2
flatten/Reshapew
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’     2
Reshape_1/shape
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2
	Reshape_1k
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
šJ
š
__inference_standard_gru_35601

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢whilef
ReadVariableOpReadVariableOpbias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ż
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’	*
shrink_axis_mask2
strided_slice_1w
MatMul/ReadVariableOpReadVariableOpkernel* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*F
_output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
bodyR
while_body_35508*
condR
while_cond_35507*E
output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimeø
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity·

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:’’’’’’’’’2

Identity_1²

Identity_2Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_2¢

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*K
_input_shapes:
8:’’’’’’’’’	:’’’’’’’’’:::*<
api_implements*(gru_2f610419-9a37-4ba3-abe9-95bfa03bda4c*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Ž3

)__inference_cudnn_gru_with_fallback_35686

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis·
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes *<
api_implements*(gru_2f610419-9a37-4ba3-abe9-95bfa03bda4c*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ę
ć
>__inference_gru_layer_call_and_return_conditional_losses_35825

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
zerosĒ
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*V
_output_shapesD
B:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference_standard_gru_356012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’	:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
“
`
B__inference_dropout_layer_call_and_return_conditional_losses_36342

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:’’’’’’’’’	2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
½
§
&__inference_conv2d_layer_call_fn_29662

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_296542
StatefulPartitionedCallØ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs


while_cond_29303
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice-
)while_cond_29303___redundant_placeholder0-
)while_cond_29303___redundant_placeholder1-
)while_cond_29303___redundant_placeholder2-
)while_cond_29303___redundant_placeholder3-
)while_cond_29303___redundant_placeholder4
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2: : : : :’’’’’’’’’: :::::
ē
Ł
@__inference_dense_layer_call_and_return_conditional_losses_32928

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs


while_cond_33730
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice-
)while_cond_33730___redundant_placeholder0-
)while_cond_33730___redundant_placeholder1-
)while_cond_33730___redundant_placeholder2-
)while_cond_33730___redundant_placeholder3-
)while_cond_33730___redundant_placeholder4
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2: : : : :’’’’’’’’’: :::::


while_cond_31130
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice-
)while_cond_31130___redundant_placeholder0-
)while_cond_31130___redundant_placeholder1-
)while_cond_31130___redundant_placeholder2-
)while_cond_31130___redundant_placeholder3-
)while_cond_31130___redundant_placeholder4
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2: : : : :’’’’’’’’’: :::::
č
Ü
C__inference_conv2d_1_layer_call_and_return_conditional_losses_29867

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
­

M__inference_time_distributed_4_layer_call_and_return_conditional_losses_30162

inputs
identity¢dropout/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
Reshapeē
dropout/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_300982!
dropout/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape¬
	Reshape_1Reshape(dropout/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2
	Reshape_1
IdentityIdentityReshape_1:output:0 ^dropout/StatefulPartitionedCall*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’	2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
¦
ź
:__inference___backward_cudnn_gru_with_fallback_32332_32468
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeę
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationą
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĒ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:’’’’’’’’’2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::’’’’’’’’’	:’’’’’’’’’: :*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation÷
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’	2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeė
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_1_grad/ShapeÉ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_2_grad/ShapeĖ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_3_grad/ShapeĖ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/ShapeĖ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/ShapeĖ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/ShapeĖ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/ShapeĘ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/ShapeĘ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/ShapeĘ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/ShapeÉ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/ShapeŹ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/ShapeŹ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationį
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationį
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationį
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationį
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationį
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationį
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:2
gradients/split_2_grad/concat„
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
	2
gradients/split_grad/concat­
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
gradients/Reshape_grad/ShapeÄ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	2 
gradients/Reshape_grad/ReshapeŖ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:’’’’’’’’’	2

Identity®

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1 

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
	2

Identity_2¢

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
2

Identity_3¢

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*„
_input_shapes
:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: :’’’’’’’’’:::::’’’’’’’’’: ::’’’’’’’’’	:’’’’’’’’’: :::’’’’’’’’’: ::::::: : : *<
api_implements*(gru_398d159f-3a18-4e10-83f6-a767482eb326*
api_preferred_deviceGPU*B
forward_function_name)'__forward_cudnn_gru_with_fallback_324672T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
Ŗ
i
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_31995

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’,   0      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02	
Reshape¼
max_pooling2d_1/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:’’’’’’’’’	*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2
Reshape_1/shape
	Reshape_1Reshape max_pooling2d_1/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’,0:& "
 
_user_specified_nameinputs
”
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_32905

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:’’’’’’’’’2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:’’’’’’’’’:& "
 
_user_specified_nameinputs

N
2__inference_time_distributed_1_layer_call_fn_34238

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’/3 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_319192
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:’’’’’’’’’/3 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’ė :& "
 
_user_specified_nameinputs
­
±
0__inference_time_distributed_layer_call_fn_34210

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*5
_output_shapes#
!:’’’’’’’’’ė *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_318842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:’’’’’’’’’ė 2

Identity"
identityIdentity:output:0*<
_input_shapes+
):’’’’’’’’’ī::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

^
B__inference_flatten_layer_call_and_return_conditional_losses_30205

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
²
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_29764

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:& "
 
_user_specified_nameinputs
@
ą
'__forward_cudnn_gru_with_fallback_35417

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *<
api_implements*(gru_65b137f1-510e-4453-a02f-9d4932b073b5*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_cudnn_gru_with_fallback_35282_354182
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Ž3

)__inference_cudnn_gru_with_fallback_31706

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis·
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes *<
api_implements*(gru_2403a229-d2f7-4f2b-b90b-fc7205d3113b*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
šJ
š
__inference_standard_gru_29397

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢whilef
ReadVariableOpReadVariableOpbias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ż
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’	*
shrink_axis_mask2
strided_slice_1w
MatMul/ReadVariableOpReadVariableOpkernel* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*F
_output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
bodyR
while_body_29304*
condR
while_cond_29303*E
output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimeø
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity·

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:’’’’’’’’’2

Identity_1²

Identity_2Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_2¢

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*K
_input_shapes:
8:’’’’’’’’’	:’’’’’’’’’:::*<
api_implements*(gru_4417fa00-1c49-4e6a-ba14-dc50117fc1d4*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias

N
2__inference_time_distributed_4_layer_call_fn_34507

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_320392
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
Č
±
0__inference_time_distributed_layer_call_fn_34159

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*>
_output_shapes,
*:(’’’’’’’’’’’’’’’’’’ė *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_297262
StatefulPartitionedCall„
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*>
_output_shapes,
*:(’’’’’’’’’’’’’’’’’’ė 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:(’’’’’’’’’’’’’’’’’’ī::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs


K__inference_time_distributed_layer_call_and_return_conditional_losses_31869

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ī        2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ī2	
ReshapeŖ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpÅ
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė *
paddingVALID*
strides
2
conv2d/Conv2D”
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¦
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
conv2d/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ė         2
Reshape_1/shape
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ė 2
	Reshape_1³
IdentityIdentityReshape_1:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*5
_output_shapes#
!:’’’’’’’’’ė 2

Identity"
identityIdentity:output:0*<
_input_shapes+
):’’’’’’’’’ī::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
¦
ź
:__inference___backward_cudnn_gru_with_fallback_36076_36212
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeę
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationą
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĒ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:’’’’’’’’’2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::’’’’’’’’’	:’’’’’’’’’: :*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation÷
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’	2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeė
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_1_grad/ShapeÉ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_2_grad/ShapeĖ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_3_grad/ShapeĖ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/ShapeĖ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/ShapeĖ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/ShapeĖ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/ShapeĘ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/ShapeĘ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/ShapeĘ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/ShapeÉ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/ShapeŹ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/ShapeŹ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationį
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationį
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationį
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationį
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationį
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationį
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:2
gradients/split_2_grad/concat„
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
	2
gradients/split_grad/concat­
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
gradients/Reshape_grad/ShapeÄ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	2 
gradients/Reshape_grad/ReshapeŖ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:’’’’’’’’’	2

Identity®

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1 

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
	2

Identity_2¢

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
2

Identity_3¢

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*„
_input_shapes
:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: :’’’’’’’’’:::::’’’’’’’’’: ::’’’’’’’’’	:’’’’’’’’’: :::’’’’’’’’’: ::::::: : : *<
api_implements*(gru_3199de74-87ed-47a3-b6a7-759f55baaff7*
api_preferred_deviceGPU*B
forward_function_name)'__forward_cudnn_gru_with_fallback_362112T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
Ź

K__inference_time_distributed_layer_call_and_return_conditional_losses_34128

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ī        2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ī2	
ReshapeŖ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpÅ
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė *
paddingVALID*
strides
2
conv2d/Conv2D”
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¦
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
conv2d/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ė2
Reshape_1/shape/2i
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*>
_output_shapes,
*:(’’’’’’’’’’’’’’’’’’ė 2
	Reshape_1¼
IdentityIdentityReshape_1:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*>
_output_shapes,
*:(’’’’’’’’’’’’’’’’’’ė 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:(’’’’’’’’’’’’’’’’’’ī::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
«
³
2__inference_time_distributed_2_layer_call_fn_34383

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’,0*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_319452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:’’’’’’’’’,02

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’/3 ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

i
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_34602

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
flatten/Const
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2
flatten/Reshapeq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :	2
Reshape_1/shape/2Ø
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2
	Reshape_1t
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’	:& "
 
_user_specified_nameinputs
šJ
š
__inference_standard_gru_32246

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢whilef
ReadVariableOpReadVariableOpbias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ż
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’	*
shrink_axis_mask2
strided_slice_1w
MatMul/ReadVariableOpReadVariableOpkernel* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*F
_output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
bodyR
while_body_32153*
condR
while_cond_32152*E
output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimeø
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity·

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:’’’’’’’’’2

Identity_1²

Identity_2Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_2¢

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*K
_input_shapes:
8:’’’’’’’’’	:’’’’’’’’’:::*<
api_implements*(gru_398d159f-3a18-4e10-83f6-a767482eb326*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
“u
	
E__inference_sequential_layer_call_and_return_conditional_losses_34068

inputs:
6time_distributed_conv2d_conv2d_readvariableop_resource;
7time_distributed_conv2d_biasadd_readvariableop_resource>
:time_distributed_2_conv2d_1_conv2d_readvariableop_resource?
;time_distributed_2_conv2d_1_biasadd_readvariableop_resource&
"gru_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_3&
"gru_statefulpartitionedcall_args_4(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢gru/StatefulPartitionedCall¢.time_distributed/conv2d/BiasAdd/ReadVariableOp¢-time_distributed/conv2d/Conv2D/ReadVariableOp¢2time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp¢1time_distributed_2/conv2d_1/Conv2D/ReadVariableOp
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ī        2 
time_distributed/Reshape/shape¬
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ī2
time_distributed/ReshapeŻ
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-time_distributed/conv2d/Conv2D/ReadVariableOp
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė *
paddingVALID*
strides
2 
time_distributed/conv2d/Conv2DŌ
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.time_distributed/conv2d/BiasAdd/ReadVariableOpź
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2!
time_distributed/conv2d/BiasAddŖ
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
time_distributed/conv2d/Relu”
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ė         2"
 time_distributed/Reshape_1/shapeŚ
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ė 2
time_distributed/Reshape_1
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ė         2"
 time_distributed_1/Reshape/shapeĻ
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
time_distributed_1/Reshapeń
(time_distributed_1/max_pooling2d/MaxPoolMaxPool#time_distributed_1/Reshape:output:0*/
_output_shapes
:’’’’’’’’’/3 *
ksize
*
paddingVALID*
strides
2*
(time_distributed_1/max_pooling2d/MaxPool„
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   /   3       2$
"time_distributed_1/Reshape_1/shapeå
time_distributed_1/Reshape_1Reshape1time_distributed_1/max_pooling2d/MaxPool:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’/3 2
time_distributed_1/Reshape_1
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’/   3       2"
 time_distributed_2/Reshape/shapeĻ
time_distributed_2/ReshapeReshape%time_distributed_1/Reshape_1:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’/3 2
time_distributed_2/Reshapeé
1time_distributed_2/conv2d_1/Conv2D/ReadVariableOpReadVariableOp:time_distributed_2_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1time_distributed_2/conv2d_1/Conv2D/ReadVariableOp
"time_distributed_2/conv2d_1/Conv2DConv2D#time_distributed_2/Reshape:output:09time_distributed_2/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,0*
paddingVALID*
strides
2$
"time_distributed_2/conv2d_1/Conv2Dą
2time_distributed_2/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp;time_distributed_2_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2time_distributed_2/conv2d_1/BiasAdd/ReadVariableOpų
#time_distributed_2/conv2d_1/BiasAddBiasAdd+time_distributed_2/conv2d_1/Conv2D:output:0:time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,02%
#time_distributed_2/conv2d_1/BiasAdd“
 time_distributed_2/conv2d_1/ReluRelu,time_distributed_2/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02"
 time_distributed_2/conv2d_1/Relu„
"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ,   0      2$
"time_distributed_2/Reshape_1/shapeā
time_distributed_2/Reshape_1Reshape.time_distributed_2/conv2d_1/Relu:activations:0+time_distributed_2/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’,02
time_distributed_2/Reshape_1
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’,   0      2"
 time_distributed_3/Reshape/shapeĻ
time_distributed_3/ReshapeReshape%time_distributed_2/Reshape_1:output:0)time_distributed_3/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02
time_distributed_3/Reshapeõ
*time_distributed_3/max_pooling2d_1/MaxPoolMaxPool#time_distributed_3/Reshape:output:0*/
_output_shapes
:’’’’’’’’’	*
ksize
*
paddingVALID*
strides
2,
*time_distributed_3/max_pooling2d_1/MaxPool„
"time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2$
"time_distributed_3/Reshape_1/shapeē
time_distributed_3/Reshape_1Reshape3time_distributed_3/max_pooling2d_1/MaxPool:output:0+time_distributed_3/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2
time_distributed_3/Reshape_1
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2"
 time_distributed_4/Reshape/shapeĻ
time_distributed_4/ReshapeReshape%time_distributed_3/Reshape_1:output:0)time_distributed_4/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
time_distributed_4/Reshapeµ
#time_distributed_4/dropout/IdentityIdentity#time_distributed_4/Reshape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2%
#time_distributed_4/dropout/Identity„
"time_distributed_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2$
"time_distributed_4/Reshape_1/shapeą
time_distributed_4/Reshape_1Reshape,time_distributed_4/dropout/Identity:output:0+time_distributed_4/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2
time_distributed_4/Reshape_1
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2"
 time_distributed_5/Reshape/shapeĻ
time_distributed_5/ReshapeReshape%time_distributed_4/Reshape_1:output:0)time_distributed_5/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
time_distributed_5/Reshape
 time_distributed_5/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2"
 time_distributed_5/flatten/ConstÖ
"time_distributed_5/flatten/ReshapeReshape#time_distributed_5/Reshape:output:0)time_distributed_5/flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2$
"time_distributed_5/flatten/Reshape
"time_distributed_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’     2$
"time_distributed_5/Reshape_1/shapeŲ
time_distributed_5/Reshape_1Reshape+time_distributed_5/flatten/Reshape:output:0+time_distributed_5/Reshape_1/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2
time_distributed_5/Reshape_1k
	gru/ShapeShape%time_distributed_5/Reshape_1:output:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2ś
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slicee
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessk
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
gru/zeros/packed/1
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
	gru/zerosž
gru/StatefulPartitionedCallStatefulPartitionedCall%time_distributed_5/Reshape_1:output:0gru/zeros:output:0"gru_statefulpartitionedcall_args_2"gru_statefulpartitionedcall_args_3"gru_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*V
_output_shapesD
B:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference_standard_gru_338242
gru/StatefulPartitionedCall
dropout_1/IdentityIdentity$gru/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_1/Identity 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	<*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout_1/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dense/BiasAdd„
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:<(*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’(2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_1/BiasAdd/ReadVariableOp”
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’(2
dense_1/BiasAdd„
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/BiasAdd:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp”
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_2/Sigmoid
IdentityIdentitydense_2/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^gru/StatefulPartitionedCall/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp3^time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp2^time_distributed_2/conv2d_1/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:’’’’’’’’’ī:::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2`
.time_distributed/conv2d/BiasAdd/ReadVariableOp.time_distributed/conv2d/BiasAdd/ReadVariableOp2^
-time_distributed/conv2d/Conv2D/ReadVariableOp-time_distributed/conv2d/Conv2D/ReadVariableOp2h
2time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp2time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp2f
1time_distributed_2/conv2d_1/Conv2D/ReadVariableOp1time_distributed_2/conv2d_1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
ä.
Ā
while_body_33731
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstack¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  23
1TensorArrayV2Read/TensorListGetItem/element_shape¶
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’	*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMulu
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1}
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanhd
mul_1MulSigmoid:y:0placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3µ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5~
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2­

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_4"(
biasadd_1_unstackbiasadd_1_unstack_0"$
biasadd_unstackbiasadd_unstack_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*I
_input_shapes8
6: : : : :’’’’’’’’’: : ::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
Č
Ź
#__inference_gru_layer_call_fn_35428
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_314482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:’’’’’’’’’’’’’’’’’’	:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
ā
i
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_34497

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
Reshape|
dropout/IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/Identity
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2
Reshape_1/shape
	Reshape_1Reshapedropout/Identity:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
ž
N
2__inference_time_distributed_5_layer_call_fn_34642

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_320702
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’	:& "
 
_user_specified_nameinputs

i
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_29851

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ė         2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2	
Reshapeį
max_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:’’’’’’’’’/3 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_297642
max_pooling2d/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :/2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :32
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeŖ
	Reshape_1Reshape&max_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’/3 2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’/3 2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:(’’’’’’’’’’’’’’’’’’ė :& "
 
_user_specified_nameinputs
ņ¦
ź
:__inference___backward_cudnn_gru_with_fallback_31310_31446
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_0
gradients/grad_ys_1Identityplaceholder_1*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeļ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationé
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĒ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*W
_output_shapesE
C:’’’’’’’’’’’’’’’’’’	:’’’’’’’’’: :*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeė
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_1_grad/ShapeÉ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_2_grad/ShapeĖ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_3_grad/ShapeĖ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/ShapeĖ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/ShapeĖ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/ShapeĖ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/ShapeĘ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/ShapeĘ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/ShapeĘ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/ShapeÉ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/ShapeŹ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/ShapeŹ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationį
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationį
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationį
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationį
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationį
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationį
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:2
gradients/split_2_grad/concat„
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
	2
gradients/split_grad/concat­
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
gradients/Reshape_grad/ShapeÄ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	2 
gradients/Reshape_grad/Reshape³
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2

Identity®

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1 

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
	2

Identity_2¢

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
2

Identity_3¢

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*Ą
_input_shapes®
«:’’’’’’’’’:’’’’’’’’’’’’’’’’’’:’’’’’’’’’: :’’’’’’’’’’’’’’’’’’:::::’’’’’’’’’: ::’’’’’’’’’’’’’’’’’’	:’’’’’’’’’: :::’’’’’’’’’: ::::::: : : *<
api_implements*(gru_2b000e56-d911-4bed-81ad-c1f1a0027215*
api_preferred_deviceGPU*B
forward_function_name)'__forward_cudnn_gru_with_fallback_314452T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
ź
C
'__inference_dropout_layer_call_fn_36352

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_301032
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’	:& "
 
_user_specified_nameinputs


while_cond_35896
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice-
)while_cond_35896___redundant_placeholder0-
)while_cond_35896___redundant_placeholder1-
)while_cond_35896___redundant_placeholder2-
)while_cond_35896___redundant_placeholder3-
)while_cond_35896___redundant_placeholder4
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2: : : : :’’’’’’’’’: :::::


K__inference_time_distributed_layer_call_and_return_conditional_losses_34196

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ī        2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ī2	
ReshapeŖ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpÅ
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė *
paddingVALID*
strides
2
conv2d/Conv2D”
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¦
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
conv2d/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ė         2
Reshape_1/shape
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ė 2
	Reshape_1³
IdentityIdentityReshape_1:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*5
_output_shapes#
!:’’’’’’’’’ė 2

Identity"
identityIdentity:output:0*<
_input_shapes+
):’’’’’’’’’ī::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
šJ
š
__inference_standard_gru_35990

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢whilef
ReadVariableOpReadVariableOpbias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ż
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’	*
shrink_axis_mask2
strided_slice_1w
MatMul/ReadVariableOpReadVariableOpkernel* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*F
_output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
bodyR
while_body_35897*
condR
while_cond_35896*E
output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimeø
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity·

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:’’’’’’’’’2

Identity_1²

Identity_2Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_2¢

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*K
_input_shapes:
8:’’’’’’’’’	:’’’’’’’’’:::*<
api_implements*(gru_3199de74-87ed-47a3-b6a7-759f55baaff7*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
å7
ä
E__inference_sequential_layer_call_and_return_conditional_losses_32986
input_13
/time_distributed_statefulpartitionedcall_args_13
/time_distributed_statefulpartitionedcall_args_25
1time_distributed_2_statefulpartitionedcall_args_15
1time_distributed_2_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_1&
"gru_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢(time_distributed/StatefulPartitionedCall¢*time_distributed_2/StatefulPartitionedCall¢*time_distributed_4/StatefulPartitionedCallć
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinput_1/time_distributed_statefulpartitionedcall_args_1/time_distributed_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*5
_output_shapes#
!:’’’’’’’’’ė *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_318692*
(time_distributed/StatefulPartitionedCall
"time_distributed_1/PartitionedCallPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’/3 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_319102$
"time_distributed_1/PartitionedCall
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_1/PartitionedCall:output:01time_distributed_2_statefulpartitionedcall_args_11time_distributed_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’,0*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_319452,
*time_distributed_2/StatefulPartitionedCall
"time_distributed_3/PartitionedCallPartitionedCall3time_distributed_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_319862$
"time_distributed_3/PartitionedCall§
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_320302,
*time_distributed_4/StatefulPartitionedCall
"time_distributed_5/PartitionedCallPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_320602$
"time_distributed_5/PartitionedCallŽ
gru/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_5/PartitionedCall:output:0"gru_statefulpartitionedcall_args_1"gru_statefulpartitionedcall_args_2"gru_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_324702
gru/StatefulPartitionedCall§
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0+^time_distributed_4/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_329002#
!dropout_1/StatefulPartitionedCallĮ
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’<*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_329282
dense/StatefulPartitionedCallĒ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’(*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_329502!
dense_1/StatefulPartitionedCallÉ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_329732!
dense_2/StatefulPartitionedCall§
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^gru/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall+^time_distributed_4/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:’’’’’’’’’ī:::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
ķ
i
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_34408

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’,   0      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02	
Reshape¼
max_pooling2d_1/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:’’’’’’’’’	*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape¤
	Reshape_1Reshape max_pooling2d_1/MaxPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’,0:& "
 
_user_specified_nameinputs


K__inference_time_distributed_layer_call_and_return_conditional_losses_31884

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ī        2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ī2	
ReshapeŖ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpÅ
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė *
paddingVALID*
strides
2
conv2d/Conv2D”
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¦
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
conv2d/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ė         2
Reshape_1/shape
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ė 2
	Reshape_1³
IdentityIdentityReshape_1:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*5
_output_shapes#
!:’’’’’’’’’ė 2

Identity"
identityIdentity:output:0*<
_input_shapes+
):’’’’’’’’’ī::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
é$
l
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_34540

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
Reshapeq
dropout/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/dropout/raten
dropout/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape
"dropout/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dropout/dropout/random_uniform/min
"dropout/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"dropout/dropout/random_uniform/maxŌ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	*
dtype02.
,dropout/dropout/random_uniform/RandomUniformŹ
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"dropout/dropout/random_uniform/subč
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:’’’’’’’’’	2$
"dropout/dropout/random_uniform/mulÖ
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2 
dropout/dropout/random_uniforms
dropout/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/dropout/sub/x
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/dropout/sub{
dropout/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/dropout/truediv/x
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/dropout/truedivÉ
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/dropout/GreaterEqual
dropout/dropout/mulMulReshape:output:0dropout/dropout/truediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/dropout/mul
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’	2
dropout/dropout/Cast¢
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/dropout/mul_1q
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapedropout/dropout/mul_1:z:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’	:& "
 
_user_specified_nameinputs
Ŗ
N
2__inference_time_distributed_1_layer_call_fn_34284

inputs
identityĶ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’/3 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_298512
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’/3 2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:(’’’’’’’’’’’’’’’’’’ė :& "
 
_user_specified_nameinputs
Ø
N
2__inference_time_distributed_3_layer_call_fn_34431

inputs
identityĶ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_300412
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’,0:& "
 
_user_specified_nameinputs
Ę
³
2__inference_time_distributed_2_layer_call_fn_34339

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’,0*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_299392
StatefulPartitionedCall£
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’,02

Identity"
identityIdentity:output:0*C
_input_shapes2
0:&’’’’’’’’’’’’’’’’’’/3 ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
„
i
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_34558

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
Reshape|
dropout/IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/Identityq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapedropout/Identity:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’	:& "
 
_user_specified_nameinputs
ę
ć
>__inference_gru_layer_call_and_return_conditional_losses_36214

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
zerosĒ
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*V
_output_shapesD
B:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference_standard_gru_359902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’	:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ž3

)__inference_cudnn_gru_with_fallback_31309

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis·
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes *<
api_implements*(gru_2b000e56-d911-4bed-81ad-c1f1a0027215*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias


while_cond_34713
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice-
)while_cond_34713___redundant_placeholder0-
)while_cond_34713___redundant_placeholder1-
)while_cond_34713___redundant_placeholder2-
)while_cond_34713___redundant_placeholder3-
)while_cond_34713___redundant_placeholder4
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2: : : : :’’’’’’’’’: :::::
ė
é
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_29939

inputs+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2
identity¢ conv2d_1/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’/   3       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’/3 2	
Reshape¾
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:’’’’’’’’’,0*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_298672"
 conv2d_1/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :,2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :02
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape­
	Reshape_1Reshape)conv2d_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’,02
	Reshape_1
IdentityIdentityReshape_1:output:0!^conv2d_1/StatefulPartitionedCall*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’,02

Identity"
identityIdentity:output:0*C
_input_shapes2
0:&’’’’’’’’’’’’’’’’’’/3 ::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ä.
Ā
while_body_32153
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstack¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  23
1TensorArrayV2Read/TensorListGetItem/element_shape¶
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’	*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMulu
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1}
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanhd
mul_1MulSigmoid:y:0placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3µ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5~
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2­

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_4"(
biasadd_1_unstackbiasadd_1_unstack_0"$
biasadd_unstackbiasadd_unstack_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*I
_input_shapes8
6: : : : :’’’’’’’’’: : ::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
@
ą
'__forward_cudnn_gru_with_fallback_35822

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *<
api_implements*(gru_2f610419-9a37-4ba3-abe9-95bfa03bda4c*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_cudnn_gru_with_fallback_35687_358232
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ä.
Ā
while_body_29304
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstack¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  23
1TensorArrayV2Read/TensorListGetItem/element_shape¶
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’	*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMulu
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1}
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanhd
mul_1MulSigmoid:y:0placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3µ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5~
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2­

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_4"(
biasadd_1_unstackbiasadd_1_unstack_0"$
biasadd_unstackbiasadd_unstack_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*I
_input_shapes8
6: : : : :’’’’’’’’’: : ::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
@
ą
'__forward_cudnn_gru_with_fallback_31445

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *<
api_implements*(gru_2b000e56-d911-4bed-81ad-c1f1a0027215*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_cudnn_gru_with_fallback_31310_314462
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ę
ć
>__inference_gru_layer_call_and_return_conditional_losses_32859

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
zerosĒ
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*V
_output_shapesD
B:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference_standard_gru_326352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’	:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
K
š
__inference_standard_gru_31621

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢whilef
ReadVariableOpReadVariableOpbias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ż
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’	*
shrink_axis_mask2
strided_slice_1w
MatMul/ReadVariableOpReadVariableOpkernel* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*F
_output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
bodyR
while_body_31528*
condR
while_cond_31527*E
output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeņ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permÆ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimeø
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

IdentityĄ

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2

Identity_1²

Identity_2Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_2¢

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*T
_input_shapesC
A:’’’’’’’’’’’’’’’’’’	:’’’’’’’’’:::*<
api_implements*(gru_2403a229-d2f7-4f2b-b90b-fc7205d3113b*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ē
Ū
B__inference_dense_1_layer_call_and_return_conditional_losses_36292

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’(2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’(2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’(2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ž3

)__inference_cudnn_gru_with_fallback_34892

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis·
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes *<
api_implements*(gru_214693d3-9c8e-4a5e-a5f4-04ba364da9ac*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ņ

M__inference_time_distributed_2_layer_call_and_return_conditional_losses_34308

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identity¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’/   3       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’/3 2	
Reshape°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOpÉ
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,0*
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,02
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02
conv2d_1/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :,2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :02
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’,02
	Reshape_1¾
IdentityIdentityReshape_1:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’,02

Identity"
identityIdentity:output:0*C
_input_shapes2
0:&’’’’’’’’’’’’’’’’’’/3 ::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
ē
Ł
@__inference_dense_layer_call_and_return_conditional_losses_36275

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ń
Ø
'__inference_dense_1_layer_call_fn_36299

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’(*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_329502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’(2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’<::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ź

K__inference_time_distributed_layer_call_and_return_conditional_losses_34152

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ī        2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ī2	
ReshapeŖ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpÅ
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė *
paddingVALID*
strides
2
conv2d/Conv2D”
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¦
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
conv2d/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ė2
Reshape_1/shape/2i
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*>
_output_shapes,
*:(’’’’’’’’’’’’’’’’’’ė 2
	Reshape_1¼
IdentityIdentityReshape_1:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*>
_output_shapes,
*:(’’’’’’’’’’’’’’’’’’ė 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:(’’’’’’’’’’’’’’’’’’ī::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs

i
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_34585

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
flatten/Const
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2
flatten/Reshapeq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :	2
Reshape_1/shape/2Ø
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2
	Reshape_1t
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’	:& "
 
_user_specified_nameinputs
K
š
__inference_standard_gru_31224

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢whilef
ReadVariableOpReadVariableOpbias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ż
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’	*
shrink_axis_mask2
strided_slice_1w
MatMul/ReadVariableOpReadVariableOpkernel* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*F
_output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
bodyR
while_body_31131*
condR
while_cond_31130*E
output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeņ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permÆ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimeø
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

IdentityĄ

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2

Identity_1²

Identity_2Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_2¢

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*T
_input_shapesC
A:’’’’’’’’’’’’’’’’’’	:’’’’’’’’’:::*<
api_implements*(gru_2b000e56-d911-4bed-81ad-c1f1a0027215*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
@
ą
'__forward_cudnn_gru_with_fallback_31842

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *<
api_implements*(gru_2403a229-d2f7-4f2b-b90b-fc7205d3113b*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_cudnn_gru_with_fallback_31707_318432
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
@
ą
'__forward_cudnn_gru_with_fallback_34045

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *<
api_implements*(gru_610e0db3-a6a3-4ac2-b3aa-b93ab2fb5213*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_cudnn_gru_with_fallback_33910_340462
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Ž3

)__inference_cudnn_gru_with_fallback_33909

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis·
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes *<
api_implements*(gru_610e0db3-a6a3-4ac2-b3aa-b93ab2fb5213*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Å	
Ū
B__inference_dense_2_layer_call_and_return_conditional_losses_36310

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ģ
K
/__inference_max_pooling2d_1_layer_call_fn_29983

inputs
identityŲ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_299772
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:& "
 
_user_specified_nameinputs
Ü
C
'__inference_flatten_layer_call_fn_36363

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_302052
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
Ŗ
i
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_34454

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’,   0      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02	
Reshape¼
max_pooling2d_1/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:’’’’’’’’’	*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2
Reshape_1/shape
	Reshape_1Reshape max_pooling2d_1/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’,0:& "
 
_user_specified_nameinputs
 
i
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_30041

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’,   0      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02	
Reshapeē
max_pooling2d_1/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_299772!
max_pooling2d_1/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape¬
	Reshape_1Reshape(max_pooling2d_1/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’,0:& "
 
_user_specified_nameinputs
Ŗ
N
2__inference_time_distributed_1_layer_call_fn_34279

inputs
identityĶ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’/3 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_298282
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’/3 2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:(’’’’’’’’’’’’’’’’’’ė :& "
 
_user_specified_nameinputs
ē
a
B__inference_dropout_layer_call_and_return_conditional_losses_30098

inputs
identitya
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/random_uniform/max¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	*
dtype02&
$dropout/random_uniform/RandomUniformŖ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/subČ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/random_uniform/mul¶
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv©
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/mul
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’	2
dropout/Cast
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
ē
Ū
B__inference_dense_1_layer_call_and_return_conditional_losses_32950

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’(2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’(2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’(2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ø
i
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34219

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ė         2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2	
Reshapeø
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:’’’’’’’’’/3 *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   /   3       2
Reshape_1/shape
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’/3 2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:’’’’’’’’’/3 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’ė :& "
 
_user_specified_nameinputs
ä.
Ā
while_body_32542
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstack¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  23
1TensorArrayV2Read/TensorListGetItem/element_shape¶
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’	*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMulu
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1}
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanhd
mul_1MulSigmoid:y:0placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3µ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5~
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2­

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_4"(
biasadd_1_unstackbiasadd_1_unstack_0"$
biasadd_unstackbiasadd_unstack_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*I
_input_shapes8
6: : : : :’’’’’’’’’: : ::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
@
ą
'__forward_cudnn_gru_with_fallback_36211

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *<
api_implements*(gru_3199de74-87ed-47a3-b6a7-759f55baaff7*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_cudnn_gru_with_fallback_36076_362122
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ØD
É
__inference__traced_save_36486
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop6
2savev2_time_distributed_kernel_read_readvariableop4
0savev2_time_distributed_bias_read_readvariableop8
4savev2_time_distributed_2_kernel_read_readvariableop6
2savev2_time_distributed_2_bias_read_readvariableop)
%savev2_gru_kernel_read_readvariableop3
/savev2_gru_recurrent_kernel_read_readvariableop'
#savev2_gru_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_rmsprop_dense_kernel_rms_read_readvariableop5
1savev2_rmsprop_dense_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_1_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_2_bias_rms_read_readvariableopB
>savev2_rmsprop_time_distributed_kernel_rms_read_readvariableop@
<savev2_rmsprop_time_distributed_bias_rms_read_readvariableopD
@savev2_rmsprop_time_distributed_2_kernel_rms_read_readvariableopB
>savev2_rmsprop_time_distributed_2_bias_rms_read_readvariableop5
1savev2_rmsprop_gru_kernel_rms_read_readvariableop?
;savev2_rmsprop_gru_recurrent_kernel_rms_read_readvariableop3
/savev2_rmsprop_gru_bias_rms_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1„
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_73801ccde03f4745a96d2259dc32734e/part2
StringJoin/inputs_1

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename©
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*»
value±B®!B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesŹ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop2savev2_time_distributed_kernel_read_readvariableop0savev2_time_distributed_bias_read_readvariableop4savev2_time_distributed_2_kernel_read_readvariableop2savev2_time_distributed_2_bias_read_readvariableop%savev2_gru_kernel_read_readvariableop/savev2_gru_recurrent_kernel_read_readvariableop#savev2_gru_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_rmsprop_dense_kernel_rms_read_readvariableop1savev2_rmsprop_dense_bias_rms_read_readvariableop5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop3savev2_rmsprop_dense_1_bias_rms_read_readvariableop5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop3savev2_rmsprop_dense_2_bias_rms_read_readvariableop>savev2_rmsprop_time_distributed_kernel_rms_read_readvariableop<savev2_rmsprop_time_distributed_bias_rms_read_readvariableop@savev2_rmsprop_time_distributed_2_kernel_rms_read_readvariableop>savev2_rmsprop_time_distributed_2_bias_rms_read_readvariableop1savev2_rmsprop_gru_kernel_rms_read_readvariableop;savev2_rmsprop_gru_recurrent_kernel_rms_read_readvariableop/savev2_rmsprop_gru_bias_rms_read_readvariableop"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesĻ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ć
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Æ
_input_shapes
: :	<:<:<(:(:(:: : : : : : : : ::
	:
:	: : :	<:<:<(:(:(:: : : ::
	:
:	: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
Ø
i
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34228

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ė         2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2	
Reshapeø
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:’’’’’’’’’/3 *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   /   3       2
Reshape_1/shape
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’/3 2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:’’’’’’’’’/3 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’ė :& "
 
_user_specified_nameinputs
Ł
E
)__inference_dropout_1_layer_call_fn_36265

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_329052
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:& "
 
_user_specified_nameinputs
³
¼
*__inference_sequential_layer_call_fn_33061
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity¢StatefulPartitionedCallś
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_330452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:’’’’’’’’’ī:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
Ė
i
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_32070

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
flatten/Const
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2
flatten/Reshapew
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’     2
Reshape_1/shape
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2
	Reshape_1k
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’	:& "
 
_user_specified_nameinputs


while_cond_31527
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice-
)while_cond_31527___redundant_placeholder0-
)while_cond_31527___redundant_placeholder1-
)while_cond_31527___redundant_placeholder2-
)while_cond_31527___redundant_placeholder3-
)while_cond_31527___redundant_placeholder4
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2: : : : :’’’’’’’’’: :::::
ė
i
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34274

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ė         2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2	
Reshapeø
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:’’’’’’’’’/3 *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :/2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :32
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape¢
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’/3 2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’/3 2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:(’’’’’’’’’’’’’’’’’’ė :& "
 
_user_specified_nameinputs

å
>__inference_gru_layer_call_and_return_conditional_losses_35420
inputs_0"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
zerosŅ
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*_
_output_shapesM
K:’’’’’’’’’:’’’’’’’’’’’’’’’’’’:’’’’’’’’’: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference_standard_gru_351962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:’’’’’’’’’’’’’’’’’’	:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0

µ
#__inference_signature_wrapper_33134
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__wrapped_model_296412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:’’’’’’’’’ī:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1


while_cond_35102
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice-
)while_cond_35102___redundant_placeholder0-
)while_cond_35102___redundant_placeholder1-
)while_cond_35102___redundant_placeholder2-
)while_cond_35102___redundant_placeholder3-
)while_cond_35102___redundant_placeholder4
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2: : : : :’’’’’’’’’: :::::
¦
ź
:__inference___backward_cudnn_gru_with_fallback_35687_35823
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeę
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationą
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĒ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:’’’’’’’’’2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::’’’’’’’’’	:’’’’’’’’’: :*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation÷
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’	2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeė
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_1_grad/ShapeÉ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_2_grad/ShapeĖ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_3_grad/ShapeĖ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/ShapeĖ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/ShapeĖ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/ShapeĖ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/ShapeĘ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/ShapeĘ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/ShapeĘ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/ShapeÉ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/ShapeŹ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/ShapeŹ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationį
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationį
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationį
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationį
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationį
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationį
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:2
gradients/split_2_grad/concat„
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
	2
gradients/split_grad/concat­
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
gradients/Reshape_grad/ShapeÄ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	2 
gradients/Reshape_grad/ReshapeŖ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:’’’’’’’’’	2

Identity®

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1 

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
	2

Identity_2¢

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
2

Identity_3¢

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*„
_input_shapes
:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: :’’’’’’’’’:::::’’’’’’’’’: ::’’’’’’’’’	:’’’’’’’’’: :::’’’’’’’’’: ::::::: : : *<
api_implements*(gru_2f610419-9a37-4ba3-abe9-95bfa03bda4c*
api_preferred_deviceGPU*B
forward_function_name)'__forward_cudnn_gru_with_fallback_358222T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
Ž3

)__inference_cudnn_gru_with_fallback_36075

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis·
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes *<
api_implements*(gru_3199de74-87ed-47a3-b6a7-759f55baaff7*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
@
ą
'__forward_cudnn_gru_with_fallback_33578

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *<
api_implements*(gru_25539354-e5fc-47b6-a271-11b89747578c*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_cudnn_gru_with_fallback_33443_335792
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
°
»
*__inference_sequential_layer_call_fn_34104

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity¢StatefulPartitionedCallł
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_330912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:’’’’’’’’’ī:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ņ¦
ź
:__inference___backward_cudnn_gru_with_fallback_31707_31843
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_0
gradients/grad_ys_1Identityplaceholder_1*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeļ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationé
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĒ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*W
_output_shapesE
C:’’’’’’’’’’’’’’’’’’	:’’’’’’’’’: :*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeė
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_1_grad/ShapeÉ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_2_grad/ShapeĖ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_3_grad/ShapeĖ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/ShapeĖ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/ShapeĖ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/ShapeĖ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/ShapeĘ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/ShapeĘ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/ShapeĘ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/ShapeÉ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/ShapeŹ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/ShapeŹ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationį
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationį
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationį
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationį
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationį
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationį
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:2
gradients/split_2_grad/concat„
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
	2
gradients/split_grad/concat­
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
gradients/Reshape_grad/ShapeÄ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	2 
gradients/Reshape_grad/Reshape³
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2

Identity®

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1 

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
	2

Identity_2¢

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
2

Identity_3¢

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*Ą
_input_shapes®
«:’’’’’’’’’:’’’’’’’’’’’’’’’’’’:’’’’’’’’’: :’’’’’’’’’’’’’’’’’’:::::’’’’’’’’’: ::’’’’’’’’’’’’’’’’’’	:’’’’’’’’’: :::’’’’’’’’’: ::::::: : : *<
api_implements*(gru_2403a229-d2f7-4f2b-b90b-fc7205d3113b*
api_preferred_deviceGPU*B
forward_function_name)'__forward_cudnn_gru_with_fallback_318422T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
ņ¦
ź
:__inference___backward_cudnn_gru_with_fallback_35282_35418
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_0
gradients/grad_ys_1Identityplaceholder_1*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeļ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationé
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĒ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*W
_output_shapesE
C:’’’’’’’’’’’’’’’’’’	:’’’’’’’’’: :*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeė
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_1_grad/ShapeÉ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_2_grad/ShapeĖ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_3_grad/ShapeĖ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/ShapeĖ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/ShapeĖ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/ShapeĖ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/ShapeĘ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/ShapeĘ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/ShapeĘ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/ShapeÉ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/ShapeŹ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/ShapeŹ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationį
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationį
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationį
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationį
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationį
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationį
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:2
gradients/split_2_grad/concat„
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
	2
gradients/split_grad/concat­
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
gradients/Reshape_grad/ShapeÄ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	2 
gradients/Reshape_grad/Reshape³
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2

Identity®

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1 

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
	2

Identity_2¢

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
2

Identity_3¢

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*Ą
_input_shapes®
«:’’’’’’’’’:’’’’’’’’’’’’’’’’’’:’’’’’’’’’: :’’’’’’’’’’’’’’’’’’:::::’’’’’’’’’: ::’’’’’’’’’’’’’’’’’’	:’’’’’’’’’: :::’’’’’’’’’: ::::::: : : *<
api_implements*(gru_65b137f1-510e-4453-a02f-9d4932b073b5*
api_preferred_deviceGPU*B
forward_function_name)'__forward_cudnn_gru_with_fallback_354172T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
šJ
š
__inference_standard_gru_32635

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢whilef
ReadVariableOpReadVariableOpbias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ż
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’	*
shrink_axis_mask2
strided_slice_1w
MatMul/ReadVariableOpReadVariableOpkernel* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*F
_output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
bodyR
while_body_32542*
condR
while_cond_32541*E
output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimeø
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity·

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:’’’’’’’’’2

Identity_1²

Identity_2Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_2¢

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*K
_input_shapes:
8:’’’’’’’’’	:’’’’’’’’’:::*<
api_implements*(gru_e83772c8-2ae3-4ece-9026-dedf2427102b*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
«
³
2__inference_time_distributed_2_layer_call_fn_34390

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’,0*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_319602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:’’’’’’’’’,02

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’/3 ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ā7
ć
E__inference_sequential_layer_call_and_return_conditional_losses_33045

inputs3
/time_distributed_statefulpartitionedcall_args_13
/time_distributed_statefulpartitionedcall_args_25
1time_distributed_2_statefulpartitionedcall_args_15
1time_distributed_2_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_1&
"gru_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢(time_distributed/StatefulPartitionedCall¢*time_distributed_2/StatefulPartitionedCall¢*time_distributed_4/StatefulPartitionedCallā
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputs/time_distributed_statefulpartitionedcall_args_1/time_distributed_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*5
_output_shapes#
!:’’’’’’’’’ė *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_318692*
(time_distributed/StatefulPartitionedCall
"time_distributed_1/PartitionedCallPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’/3 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_319102$
"time_distributed_1/PartitionedCall
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_1/PartitionedCall:output:01time_distributed_2_statefulpartitionedcall_args_11time_distributed_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’,0*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_319452,
*time_distributed_2/StatefulPartitionedCall
"time_distributed_3/PartitionedCallPartitionedCall3time_distributed_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_319862$
"time_distributed_3/PartitionedCall§
*time_distributed_4/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_320302,
*time_distributed_4/StatefulPartitionedCall
"time_distributed_5/PartitionedCallPartitionedCall3time_distributed_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_320602$
"time_distributed_5/PartitionedCallŽ
gru/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_5/PartitionedCall:output:0"gru_statefulpartitionedcall_args_1"gru_statefulpartitionedcall_args_2"gru_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_324702
gru/StatefulPartitionedCall§
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0+^time_distributed_4/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_329002#
!dropout_1/StatefulPartitionedCallĮ
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’<*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_329282
dense/StatefulPartitionedCallĒ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’(*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_329502!
dense_1/StatefulPartitionedCallÉ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_329732!
dense_2/StatefulPartitionedCall§
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^gru/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall+^time_distributed_4/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:’’’’’’’’’ī:::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall2X
*time_distributed_4/StatefulPartitionedCall*time_distributed_4/StatefulPartitionedCall:& "
 
_user_specified_nameinputs

N
2__inference_time_distributed_3_layer_call_fn_34459

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_319862
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’,0:& "
 
_user_specified_nameinputs
“
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_29977

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:& "
 
_user_specified_nameinputs
ä.
Ā
while_body_35897
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstack¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  23
1TensorArrayV2Read/TensorListGetItem/element_shape¶
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’	*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMulu
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1}
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanhd
mul_1MulSigmoid:y:0placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3µ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5~
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2­

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_4"(
biasadd_1_unstackbiasadd_1_unstack_0"$
biasadd_unstackbiasadd_unstack_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*I
_input_shapes8
6: : : : :’’’’’’’’’: : ::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp


while_cond_35507
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice-
)while_cond_35507___redundant_placeholder0-
)while_cond_35507___redundant_placeholder1-
)while_cond_35507___redundant_placeholder2-
)while_cond_35507___redundant_placeholder3-
)while_cond_35507___redundant_placeholder4
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2: : : : :’’’’’’’’’: :::::
ä.
Ā
while_body_31528
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstack¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  23
1TensorArrayV2Read/TensorListGetItem/element_shape¶
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’	*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMulu
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1}
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanhd
mul_1MulSigmoid:y:0placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3µ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5~
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2­

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_4"(
biasadd_1_unstackbiasadd_1_unstack_0"$
biasadd_unstackbiasadd_unstack_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*I
_input_shapes8
6: : : : :’’’’’’’’’: : ::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
ę
Ś
A__inference_conv2d_layer_call_and_return_conditional_losses_29654

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Į
©
(__inference_conv2d_1_layer_call_fn_29875

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_298672
StatefulPartitionedCallØ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

å
>__inference_gru_layer_call_and_return_conditional_losses_35031
inputs_0"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
zerosŅ
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*_
_output_shapesM
K:’’’’’’’’’:’’’’’’’’’’’’’’’’’’:’’’’’’’’’: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference_standard_gru_348072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:’’’’’’’’’’’’’’’’’’	:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
³
¼
*__inference_sequential_layer_call_fn_33107
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity¢StatefulPartitionedCallś
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_330912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:’’’’’’’’’ī:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
³
k
2__inference_time_distributed_4_layer_call_fn_34563

inputs
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_301622
StatefulPartitionedCall£
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’	22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

k
2__inference_time_distributed_4_layer_call_fn_34502

inputs
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_320302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’	22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

i
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_29828

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ė         2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2	
Reshapeį
max_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:’’’’’’’’’/3 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_297642
max_pooling2d/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :/2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :32
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shapeŖ
	Reshape_1Reshape&max_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’/3 2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’/3 2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:(’’’’’’’’’’’’’’’’’’ė :& "
 
_user_specified_nameinputs
ä.
Ā
while_body_34714
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstack¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  23
1TensorArrayV2Read/TensorListGetItem/element_shape¶
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’	*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMulu
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1}
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanhd
mul_1MulSigmoid:y:0placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3µ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5~
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2­

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_4"(
biasadd_1_unstackbiasadd_1_unstack_0"$
biasadd_unstackbiasadd_unstack_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*I
_input_shapes8
6: : : : :’’’’’’’’’: : ::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
Æ

M__inference_time_distributed_2_layer_call_and_return_conditional_losses_31945

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identity¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’/   3       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’/3 2	
Reshape°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOpÉ
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,0*
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,02
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02
conv2d_1/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ,   0      2
Reshape_1/shape
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’,02
	Reshape_1µ
IdentityIdentityReshape_1:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*3
_output_shapes!
:’’’’’’’’’,02

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’/3 ::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
¤
	
E__inference_sequential_layer_call_and_return_conditional_losses_33616

inputs:
6time_distributed_conv2d_conv2d_readvariableop_resource;
7time_distributed_conv2d_biasadd_readvariableop_resource>
:time_distributed_2_conv2d_1_conv2d_readvariableop_resource?
;time_distributed_2_conv2d_1_biasadd_readvariableop_resource&
"gru_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_3&
"gru_statefulpartitionedcall_args_4(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢gru/StatefulPartitionedCall¢.time_distributed/conv2d/BiasAdd/ReadVariableOp¢-time_distributed/conv2d/Conv2D/ReadVariableOp¢2time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp¢1time_distributed_2/conv2d_1/Conv2D/ReadVariableOp
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ī        2 
time_distributed/Reshape/shape¬
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ī2
time_distributed/ReshapeŻ
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-time_distributed/conv2d/Conv2D/ReadVariableOp
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė *
paddingVALID*
strides
2 
time_distributed/conv2d/Conv2DŌ
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.time_distributed/conv2d/BiasAdd/ReadVariableOpź
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2!
time_distributed/conv2d/BiasAddŖ
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
time_distributed/conv2d/Relu”
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ė         2"
 time_distributed/Reshape_1/shapeŚ
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ė 2
time_distributed/Reshape_1
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ė         2"
 time_distributed_1/Reshape/shapeĻ
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
time_distributed_1/Reshapeń
(time_distributed_1/max_pooling2d/MaxPoolMaxPool#time_distributed_1/Reshape:output:0*/
_output_shapes
:’’’’’’’’’/3 *
ksize
*
paddingVALID*
strides
2*
(time_distributed_1/max_pooling2d/MaxPool„
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   /   3       2$
"time_distributed_1/Reshape_1/shapeå
time_distributed_1/Reshape_1Reshape1time_distributed_1/max_pooling2d/MaxPool:output:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’/3 2
time_distributed_1/Reshape_1
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’/   3       2"
 time_distributed_2/Reshape/shapeĻ
time_distributed_2/ReshapeReshape%time_distributed_1/Reshape_1:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’/3 2
time_distributed_2/Reshapeé
1time_distributed_2/conv2d_1/Conv2D/ReadVariableOpReadVariableOp:time_distributed_2_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1time_distributed_2/conv2d_1/Conv2D/ReadVariableOp
"time_distributed_2/conv2d_1/Conv2DConv2D#time_distributed_2/Reshape:output:09time_distributed_2/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,0*
paddingVALID*
strides
2$
"time_distributed_2/conv2d_1/Conv2Dą
2time_distributed_2/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp;time_distributed_2_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2time_distributed_2/conv2d_1/BiasAdd/ReadVariableOpų
#time_distributed_2/conv2d_1/BiasAddBiasAdd+time_distributed_2/conv2d_1/Conv2D:output:0:time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,02%
#time_distributed_2/conv2d_1/BiasAdd“
 time_distributed_2/conv2d_1/ReluRelu,time_distributed_2/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02"
 time_distributed_2/conv2d_1/Relu„
"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ,   0      2$
"time_distributed_2/Reshape_1/shapeā
time_distributed_2/Reshape_1Reshape.time_distributed_2/conv2d_1/Relu:activations:0+time_distributed_2/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’,02
time_distributed_2/Reshape_1
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’,   0      2"
 time_distributed_3/Reshape/shapeĻ
time_distributed_3/ReshapeReshape%time_distributed_2/Reshape_1:output:0)time_distributed_3/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02
time_distributed_3/Reshapeõ
*time_distributed_3/max_pooling2d_1/MaxPoolMaxPool#time_distributed_3/Reshape:output:0*/
_output_shapes
:’’’’’’’’’	*
ksize
*
paddingVALID*
strides
2,
*time_distributed_3/max_pooling2d_1/MaxPool„
"time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2$
"time_distributed_3/Reshape_1/shapeē
time_distributed_3/Reshape_1Reshape3time_distributed_3/max_pooling2d_1/MaxPool:output:0+time_distributed_3/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2
time_distributed_3/Reshape_1
 time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2"
 time_distributed_4/Reshape/shapeĻ
time_distributed_4/ReshapeReshape%time_distributed_3/Reshape_1:output:0)time_distributed_4/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
time_distributed_4/Reshape
'time_distributed_4/dropout/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  >2)
'time_distributed_4/dropout/dropout/rate§
(time_distributed_4/dropout/dropout/ShapeShape#time_distributed_4/Reshape:output:0*
T0*
_output_shapes
:2*
(time_distributed_4/dropout/dropout/Shape³
5time_distributed_4/dropout/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5time_distributed_4/dropout/dropout/random_uniform/min³
5time_distributed_4/dropout/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?27
5time_distributed_4/dropout/dropout/random_uniform/max
?time_distributed_4/dropout/dropout/random_uniform/RandomUniformRandomUniform1time_distributed_4/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	*
dtype02A
?time_distributed_4/dropout/dropout/random_uniform/RandomUniform
5time_distributed_4/dropout/dropout/random_uniform/subSub>time_distributed_4/dropout/dropout/random_uniform/max:output:0>time_distributed_4/dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 27
5time_distributed_4/dropout/dropout/random_uniform/sub“
5time_distributed_4/dropout/dropout/random_uniform/mulMulHtime_distributed_4/dropout/dropout/random_uniform/RandomUniform:output:09time_distributed_4/dropout/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:’’’’’’’’’	27
5time_distributed_4/dropout/dropout/random_uniform/mul¢
1time_distributed_4/dropout/dropout/random_uniformAdd9time_distributed_4/dropout/dropout/random_uniform/mul:z:0>time_distributed_4/dropout/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:’’’’’’’’’	23
1time_distributed_4/dropout/dropout/random_uniform
(time_distributed_4/dropout/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(time_distributed_4/dropout/dropout/sub/xŻ
&time_distributed_4/dropout/dropout/subSub1time_distributed_4/dropout/dropout/sub/x:output:00time_distributed_4/dropout/dropout/rate:output:0*
T0*
_output_shapes
: 2(
&time_distributed_4/dropout/dropout/sub”
,time_distributed_4/dropout/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,time_distributed_4/dropout/dropout/truediv/xē
*time_distributed_4/dropout/dropout/truedivRealDiv5time_distributed_4/dropout/dropout/truediv/x:output:0*time_distributed_4/dropout/dropout/sub:z:0*
T0*
_output_shapes
: 2,
*time_distributed_4/dropout/dropout/truediv
/time_distributed_4/dropout/dropout/GreaterEqualGreaterEqual5time_distributed_4/dropout/dropout/random_uniform:z:00time_distributed_4/dropout/dropout/rate:output:0*
T0*/
_output_shapes
:’’’’’’’’’	21
/time_distributed_4/dropout/dropout/GreaterEqualę
&time_distributed_4/dropout/dropout/mulMul#time_distributed_4/Reshape:output:0.time_distributed_4/dropout/dropout/truediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’	2(
&time_distributed_4/dropout/dropout/mulŲ
'time_distributed_4/dropout/dropout/CastCast3time_distributed_4/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’	2)
'time_distributed_4/dropout/dropout/Castī
(time_distributed_4/dropout/dropout/mul_1Mul*time_distributed_4/dropout/dropout/mul:z:0+time_distributed_4/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’	2*
(time_distributed_4/dropout/dropout/mul_1„
"time_distributed_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2$
"time_distributed_4/Reshape_1/shapeą
time_distributed_4/Reshape_1Reshape,time_distributed_4/dropout/dropout/mul_1:z:0+time_distributed_4/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2
time_distributed_4/Reshape_1
 time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2"
 time_distributed_5/Reshape/shapeĻ
time_distributed_5/ReshapeReshape%time_distributed_4/Reshape_1:output:0)time_distributed_5/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
time_distributed_5/Reshape
 time_distributed_5/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2"
 time_distributed_5/flatten/ConstÖ
"time_distributed_5/flatten/ReshapeReshape#time_distributed_5/Reshape:output:0)time_distributed_5/flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2$
"time_distributed_5/flatten/Reshape
"time_distributed_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’     2$
"time_distributed_5/Reshape_1/shapeŲ
time_distributed_5/Reshape_1Reshape+time_distributed_5/flatten/Reshape:output:0+time_distributed_5/Reshape_1/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2
time_distributed_5/Reshape_1k
	gru/ShapeShape%time_distributed_5/Reshape_1:output:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2ś
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slicee
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessk
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
gru/zeros/packed/1
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
	gru/zerosž
gru/StatefulPartitionedCallStatefulPartitionedCall%time_distributed_5/Reshape_1:output:0gru/zeros:output:0"gru_statefulpartitionedcall_args_2"gru_statefulpartitionedcall_args_3"gru_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*V
_output_shapesD
B:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference_standard_gru_333572
gru/StatefulPartitionedCallu
dropout_1/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout_1/dropout/rate
dropout_1/dropout/ShapeShape$gru/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape
$dropout_1/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$dropout_1/dropout/random_uniform/min
$dropout_1/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$dropout_1/dropout/random_uniform/maxÓ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformŅ
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2&
$dropout_1/dropout/random_uniform/subé
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:’’’’’’’’’2&
$dropout_1/dropout/random_uniform/mul×
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:’’’’’’’’’2"
 dropout_1/dropout/random_uniformw
dropout_1/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_1/dropout/sub/x
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_1/dropout/sub
dropout_1/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_1/dropout/truediv/x£
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_1/dropout/truedivŹ
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*(
_output_shapes
:’’’’’’’’’2 
dropout_1/dropout/GreaterEqual­
dropout_1/dropout/mulMul$gru/StatefulPartitionedCall:output:0dropout_1/dropout/truediv:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_1/dropout/mul
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout_1/dropout/Cast£
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout_1/dropout/mul_1 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	<*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout_1/dropout/mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
dense/BiasAdd„
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:<(*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’(2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_1/BiasAdd/ReadVariableOp”
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’(2
dense_1/BiasAdd„
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/BiasAdd:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp”
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
dense_2/Sigmoid
IdentityIdentitydense_2/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^gru/StatefulPartitionedCall/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp3^time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp2^time_distributed_2/conv2d_1/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:’’’’’’’’’ī:::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2`
.time_distributed/conv2d/BiasAdd/ReadVariableOp.time_distributed/conv2d/BiasAdd/ReadVariableOp2^
-time_distributed/conv2d/Conv2D/ReadVariableOp-time_distributed/conv2d/Conv2D/ReadVariableOp2h
2time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp2time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp2f
1time_distributed_2/conv2d_1/Conv2D/ReadVariableOp1time_distributed_2/conv2d_1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs

N
2__inference_time_distributed_5_layer_call_fn_34612

inputs
identityĘ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_302742
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’	:& "
 
_user_specified_nameinputs

^
B__inference_flatten_layer_call_and_return_conditional_losses_36358

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
ä.
Ā
while_body_35508
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstack¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  23
1TensorArrayV2Read/TensorListGetItem/element_shape¶
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’	*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMulu
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1}
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanhd
mul_1MulSigmoid:y:0placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3µ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5~
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2­

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_4"(
biasadd_1_unstackbiasadd_1_unstack_0"$
biasadd_unstackbiasadd_unstack_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*I
_input_shapes8
6: : : : :’’’’’’’’’: : ::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
Å	
Ū
B__inference_dense_2_layer_call_and_return_conditional_losses_32973

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs

N
2__inference_time_distributed_1_layer_call_fn_34233

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’/3 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_319102
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:’’’’’’’’’/3 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’ė :& "
 
_user_specified_nameinputs
Æ

M__inference_time_distributed_2_layer_call_and_return_conditional_losses_34361

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identity¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’/   3       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’/3 2	
Reshape°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOpÉ
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,0*
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,02
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02
conv2d_1/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ,   0      2
Reshape_1/shape
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’,02
	Reshape_1µ
IdentityIdentityReshape_1:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*3
_output_shapes!
:’’’’’’’’’,02

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’/3 ::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
šJ
š
__inference_standard_gru_33357

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢whilef
ReadVariableOpReadVariableOpbias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ż
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’	*
shrink_axis_mask2
strided_slice_1w
MatMul/ReadVariableOpReadVariableOpkernel* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*F
_output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
bodyR
while_body_33264*
condR
while_cond_33263*E
output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimeø
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity·

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:’’’’’’’’’2

Identity_1²

Identity_2Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_2¢

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*K
_input_shapes:
8:’’’’’’’’’	:’’’’’’’’’:::*<
api_implements*(gru_25539354-e5fc-47b6-a271-11b89747578c*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ā
i
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_32039

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
Reshape|
dropout/IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/Identity
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2
Reshape_1/shape
	Reshape_1Reshapedropout/Identity:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
ä.
Ā
while_body_33264
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstack¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  23
1TensorArrayV2Read/TensorListGetItem/element_shape¶
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’	*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMulu
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1}
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanhd
mul_1MulSigmoid:y:0placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3µ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5~
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2­

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_4"(
biasadd_1_unstackbiasadd_1_unstack_0"$
biasadd_unstackbiasadd_unstack_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*I
_input_shapes8
6: : : : :’’’’’’’’’: : ::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
ä.
Ā
while_body_35103
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstack¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  23
1TensorArrayV2Read/TensorListGetItem/element_shape¶
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’	*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMulu
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1}
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanhd
mul_1MulSigmoid:y:0placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3µ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5~
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2­

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_4"(
biasadd_1_unstackbiasadd_1_unstack_0"$
biasadd_unstackbiasadd_unstack_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*I
_input_shapes8
6: : : : :’’’’’’’’’: : ::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
¦
l
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_32030

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
Reshapeq
dropout/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/dropout/raten
dropout/dropout/ShapeShapeReshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape
"dropout/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dropout/dropout/random_uniform/min
"dropout/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"dropout/dropout/random_uniform/maxŌ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	*
dtype02.
,dropout/dropout/random_uniform/RandomUniformŹ
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"dropout/dropout/random_uniform/subč
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:’’’’’’’’’	2$
"dropout/dropout/random_uniform/mulÖ
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2 
dropout/dropout/random_uniforms
dropout/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/dropout/sub/x
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/dropout/sub{
dropout/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/dropout/truediv/x
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/dropout/truedivÉ
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/dropout/GreaterEqual
dropout/dropout/mulMulReshape:output:0dropout/dropout/truediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/dropout/mul
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’	2
dropout/dropout/Cast¢
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/dropout/mul_1
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2
Reshape_1/shape
	Reshape_1Reshapedropout/dropout/mul_1:z:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
ė
é
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_29966

inputs+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2
identity¢ conv2d_1/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’/   3       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’/3 2	
Reshape¾
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:’’’’’’’’’,0*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_298672"
 conv2d_1/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :,2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :02
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape­
	Reshape_1Reshape)conv2d_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’,02
	Reshape_1
IdentityIdentityReshape_1:output:0!^conv2d_1/StatefulPartitionedCall*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’,02

Identity"
identityIdentity:output:0*C
_input_shapes2
0:&’’’’’’’’’’’’’’’’’’/3 ::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs


while_cond_32152
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice-
)while_cond_32152___redundant_placeholder0-
)while_cond_32152___redundant_placeholder1-
)while_cond_32152___redundant_placeholder2-
)while_cond_32152___redundant_placeholder3-
)while_cond_32152___redundant_placeholder4
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2: : : : :’’’’’’’’’: :::::
ž
N
2__inference_time_distributed_5_layer_call_fn_34637

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_320602
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
Ŗ
i
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_31986

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’,   0      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02	
Reshape¼
max_pooling2d_1/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:’’’’’’’’’	*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2
Reshape_1/shape
	Reshape_1Reshape max_pooling2d_1/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’,0:& "
 
_user_specified_nameinputs
ņ¦
ź
:__inference___backward_cudnn_gru_with_fallback_34893_35029
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_0
gradients/grad_ys_1Identityplaceholder_1*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeļ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationé
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĒ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*W
_output_shapesE
C:’’’’’’’’’’’’’’’’’’	:’’’’’’’’’: :*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeė
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_1_grad/ShapeÉ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_2_grad/ShapeĖ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_3_grad/ShapeĖ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/ShapeĖ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/ShapeĖ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/ShapeĖ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/ShapeĘ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/ShapeĘ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/ShapeĘ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/ShapeÉ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/ShapeŹ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/ShapeŹ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationį
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationį
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationį
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationį
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationį
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationį
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:2
gradients/split_2_grad/concat„
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
	2
gradients/split_grad/concat­
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
gradients/Reshape_grad/ShapeÄ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	2 
gradients/Reshape_grad/Reshape³
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2

Identity®

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1 

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
	2

Identity_2¢

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
2

Identity_3¢

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*Ą
_input_shapes®
«:’’’’’’’’’:’’’’’’’’’’’’’’’’’’:’’’’’’’’’: :’’’’’’’’’’’’’’’’’’:::::’’’’’’’’’: ::’’’’’’’’’’’’’’’’’’	:’’’’’’’’’: :::’’’’’’’’’: ::::::: : : *<
api_implements*(gru_214693d3-9c8e-4a5e-a5f4-04ba364da9ac*
api_preferred_deviceGPU*B
forward_function_name)'__forward_cudnn_gru_with_fallback_350282T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
Ž3

)__inference_cudnn_gru_with_fallback_32331

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis·
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes *<
api_implements*(gru_398d159f-3a18-4e10-83f6-a767482eb326*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
¦
ź
:__inference___backward_cudnn_gru_with_fallback_33443_33579
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeę
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationą
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĒ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:’’’’’’’’’2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::’’’’’’’’’	:’’’’’’’’’: :*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation÷
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’	2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeė
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_1_grad/ShapeÉ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_2_grad/ShapeĖ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_3_grad/ShapeĖ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/ShapeĖ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/ShapeĖ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/ShapeĖ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/ShapeĘ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/ShapeĘ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/ShapeĘ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/ShapeÉ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/ShapeŹ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/ShapeŹ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationį
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationį
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationį
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationį
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationį
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationį
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:2
gradients/split_2_grad/concat„
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
	2
gradients/split_grad/concat­
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
gradients/Reshape_grad/ShapeÄ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	2 
gradients/Reshape_grad/ReshapeŖ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:’’’’’’’’’	2

Identity®

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1 

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
	2

Identity_2¢

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
2

Identity_3¢

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*„
_input_shapes
:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: :’’’’’’’’’:::::’’’’’’’’’: ::’’’’’’’’’	:’’’’’’’’’: :::’’’’’’’’’: ::::::: : : *<
api_implements*(gru_25539354-e5fc-47b6-a271-11b89747578c*
api_preferred_deviceGPU*B
forward_function_name)'__forward_cudnn_gru_with_fallback_335782T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
°
»
*__inference_sequential_layer_call_fn_34086

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity¢StatefulPartitionedCallł
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_330452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:’’’’’’’’’ī:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Æ

M__inference_time_distributed_2_layer_call_and_return_conditional_losses_34376

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identity¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’/   3       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’/3 2	
Reshape°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOpÉ
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,0*
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,02
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02
conv2d_1/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ,   0      2
Reshape_1/shape
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’,02
	Reshape_1µ
IdentityIdentityReshape_1:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*3
_output_shapes!
:’’’’’’’’’,02

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’/3 ::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Ø
N
2__inference_time_distributed_4_layer_call_fn_34568

inputs
identityĶ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_301852
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’	:& "
 
_user_specified_nameinputs
©
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_36250

inputs
identitya
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/random_uniform/maxµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’*
dtype02&
$dropout/random_uniform/RandomUniformŖ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/subĮ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/random_uniform/mulÆ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv¢
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/mul
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:’’’’’’’’’2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:& "
 
_user_specified_nameinputs
¹
Č
#__inference_gru_layer_call_fn_36222

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity¢StatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_324702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’	:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Æ

M__inference_time_distributed_2_layer_call_and_return_conditional_losses_31960

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identity¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’/   3       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’/3 2	
Reshape°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOpÉ
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,0*
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,02
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02
conv2d_1/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ,   0      2
Reshape_1/shape
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’,02
	Reshape_1µ
IdentityIdentityReshape_1:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*3
_output_shapes!
:’’’’’’’’’,02

Identity"
identityIdentity:output:0*:
_input_shapes)
':’’’’’’’’’/3 ::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Ž3

)__inference_cudnn_gru_with_fallback_33442

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis·
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes *<
api_implements*(gru_25539354-e5fc-47b6-a271-11b89747578c*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ö
`
'__inference_dropout_layer_call_fn_36347

inputs
identity¢StatefulPartitionedCallĶ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_300982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’	22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
¹
Č
#__inference_gru_layer_call_fn_36230

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity¢StatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_328592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:’’’’’’’’’	:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ņ

M__inference_time_distributed_2_layer_call_and_return_conditional_losses_34332

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identity¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’/   3       2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’/3 2	
Reshape°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOpÉ
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,0*
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,02
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02
conv2d_1/Reluq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :,2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :02
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’,02
	Reshape_1¾
IdentityIdentityReshape_1:output:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’,02

Identity"
identityIdentity:output:0*C
_input_shapes2
0:&’’’’’’’’’’’’’’’’’’/3 ::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
šJ
š
__inference_standard_gru_33824

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢whilef
ReadVariableOpReadVariableOpbias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ż
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’	*
shrink_axis_mask2
strided_slice_1w
MatMul/ReadVariableOpReadVariableOpkernel* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*F
_output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
bodyR
while_body_33731*
condR
while_cond_33730*E
output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimeø
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity·

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:’’’’’’’’’2

Identity_1²

Identity_2Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_2¢

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*K
_input_shapes:
8:’’’’’’’’’	:’’’’’’’’’:::*<
api_implements*(gru_610e0db3-a6a3-4ac2-b3aa-b93ab2fb5213*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
¦
ź
:__inference___backward_cudnn_gru_with_fallback_32721_32857
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeę
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationą
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĒ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:’’’’’’’’’2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::’’’’’’’’’	:’’’’’’’’’: :*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation÷
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’	2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeė
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_1_grad/ShapeÉ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_2_grad/ShapeĖ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_3_grad/ShapeĖ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/ShapeĖ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/ShapeĖ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/ShapeĖ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/ShapeĘ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/ShapeĘ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/ShapeĘ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/ShapeÉ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/ShapeŹ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/ShapeŹ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationį
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationį
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationį
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationį
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationį
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationį
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:2
gradients/split_2_grad/concat„
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
	2
gradients/split_grad/concat­
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
gradients/Reshape_grad/ShapeÄ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	2 
gradients/Reshape_grad/ReshapeŖ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:’’’’’’’’’	2

Identity®

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1 

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
	2

Identity_2¢

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
2

Identity_3¢

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*„
_input_shapes
:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: :’’’’’’’’’:::::’’’’’’’’’: ::’’’’’’’’’	:’’’’’’’’’: :::’’’’’’’’’: ::::::: : : *<
api_implements*(gru_e83772c8-2ae3-4ece-9026-dedf2427102b*
api_preferred_deviceGPU*B
forward_function_name)'__forward_cudnn_gru_with_fallback_328562T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop

i
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_30185

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
ReshapeĻ
dropout/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_301032
dropout/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape¤
	Reshape_1Reshape dropout/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’	:& "
 
_user_specified_nameinputs
ä.
Ā
while_body_31131
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstack¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp·
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  23
1TensorArrayV2Read/TensorListGetItem/element_shape¶
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:’’’’’’’’’	*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMulu
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1}
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanhd
mul_1MulSigmoid:y:0placeholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3µ
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5~
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2­

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_4"(
biasadd_1_unstackbiasadd_1_unstack_0"$
biasadd_unstackbiasadd_unstack_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*I
_input_shapes8
6: : : : :’’’’’’’’’: : ::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
ķ
i
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_34426

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’,   0      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02	
Reshape¼
max_pooling2d_1/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:’’’’’’’’’	*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape¤
	Reshape_1Reshape max_pooling2d_1/MaxPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’,0:& "
 
_user_specified_nameinputs
¦
ź
:__inference___backward_cudnn_gru_with_fallback_29483_29619
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeę
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationą
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĒ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:’’’’’’’’’2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::’’’’’’’’’	:’’’’’’’’’: :*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation÷
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’	2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeė
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_1_grad/ShapeÉ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_2_grad/ShapeĖ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_3_grad/ShapeĖ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/ShapeĖ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/ShapeĖ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/ShapeĖ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/ShapeĘ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/ShapeĘ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/ShapeĘ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/ShapeÉ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/ShapeŹ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/ShapeŹ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationį
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationį
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationį
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationį
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationį
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationį
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:2
gradients/split_2_grad/concat„
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
	2
gradients/split_grad/concat­
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
gradients/Reshape_grad/ShapeÄ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	2 
gradients/Reshape_grad/ReshapeŖ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:’’’’’’’’’	2

Identity®

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1 

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
	2

Identity_2¢

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
2

Identity_3¢

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*„
_input_shapes
:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: :’’’’’’’’’:::::’’’’’’’’’: ::’’’’’’’’’	:’’’’’’’’’: :::’’’’’’’’’: ::::::: : : *<
api_implements*(gru_4417fa00-1c49-4e6a-ba14-dc50117fc1d4*
api_preferred_deviceGPU*B
forward_function_name)'__forward_cudnn_gru_with_fallback_296182T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
“4

E__inference_sequential_layer_call_and_return_conditional_losses_33014
input_13
/time_distributed_statefulpartitionedcall_args_13
/time_distributed_statefulpartitionedcall_args_25
1time_distributed_2_statefulpartitionedcall_args_15
1time_distributed_2_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_1&
"gru_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢(time_distributed/StatefulPartitionedCall¢*time_distributed_2/StatefulPartitionedCallć
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinput_1/time_distributed_statefulpartitionedcall_args_1/time_distributed_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*5
_output_shapes#
!:’’’’’’’’’ė *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_318842*
(time_distributed/StatefulPartitionedCall
"time_distributed_1/PartitionedCallPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’/3 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_319192$
"time_distributed_1/PartitionedCall
*time_distributed_2/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_1/PartitionedCall:output:01time_distributed_2_statefulpartitionedcall_args_11time_distributed_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’,0*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_319602,
*time_distributed_2/StatefulPartitionedCall
"time_distributed_3/PartitionedCallPartitionedCall3time_distributed_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_319952$
"time_distributed_3/PartitionedCall
"time_distributed_4/PartitionedCallPartitionedCall+time_distributed_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_320392$
"time_distributed_4/PartitionedCall
"time_distributed_5/PartitionedCallPartitionedCall+time_distributed_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_320702$
"time_distributed_5/PartitionedCallŽ
gru/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_5/PartitionedCall:output:0"gru_statefulpartitionedcall_args_1"gru_statefulpartitionedcall_args_2"gru_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_328592
gru/StatefulPartitionedCallā
dropout_1/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_329052
dropout_1/PartitionedCall¹
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’<*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_329282
dense/StatefulPartitionedCallĒ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’(*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_329502!
dense_1/StatefulPartitionedCallÉ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_329732!
dense_2/StatefulPartitionedCallÖ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_2/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:’’’’’’’’’ī:::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_2/StatefulPartitionedCall*time_distributed_2/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
Ø
i
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_31919

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ė         2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2	
Reshapeø
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:’’’’’’’’’/3 *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   /   3       2
Reshape_1/shape
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’/3 2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:’’’’’’’’’/3 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:’’’’’’’’’ė :& "
 
_user_specified_nameinputs
K
š
__inference_standard_gru_35196

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢whilef
ReadVariableOpReadVariableOpbias*
_output_shapes
:	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2æ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  27
5TensorArrayUnstack/TensorListFromTensor/element_shapeų
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ż
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’	*
shrink_axis_mask2
strided_slice_1w
MatMul/ReadVariableOpReadVariableOpkernel* 
_output_shapes
:
	*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÆ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
	BiasAdd_1T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim·
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:’’’’’’’’’2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:’’’’’’’’’2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:’’’’’’’’’2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:’’’’’’’’’2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:’’’’’’’’’2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*F
_output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
bodyR
while_body_35103*
condR
while_cond_35102*E
output_shapes4
2: : : : :’’’’’’’’’: : : :: :*
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   22
0TensorArrayV2Stack/TensorListStack/element_shapeņ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
’’’’’’’’’2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permÆ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
runtimeø
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

IdentityĄ

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’2

Identity_1²

Identity_2Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_2¢

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*T
_input_shapesC
A:’’’’’’’’’’’’’’’’’’	:’’’’’’’’’:::*<
api_implements*(gru_65b137f1-510e-4453-a02f-9d4932b073b5*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ī
¦
%__inference_dense_layer_call_fn_36282

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:’’’’’’’’’<*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_329282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’<2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
@
ą
'__forward_cudnn_gru_with_fallback_32856

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *<
api_implements*(gru_e83772c8-2ae3-4ece-9026-dedf2427102b*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_cudnn_gru_with_fallback_32721_328572
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias


K__inference_time_distributed_layer_call_and_return_conditional_losses_34181

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOpw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ī        2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ī2	
ReshapeŖ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpÅ
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė *
paddingVALID*
strides
2
conv2d/Conv2D”
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¦
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2
conv2d/Relu
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ė         2
Reshape_1/shape
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ė 2
	Reshape_1³
IdentityIdentityReshape_1:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*5
_output_shapes#
!:’’’’’’’’’ė 2

Identity"
identityIdentity:output:0*<
_input_shapes+
):’’’’’’’’’ī::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Ŗ
i
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_34445

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’,   0      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02	
Reshape¼
max_pooling2d_1/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:’’’’’’’’’	*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2
Reshape_1/shape
	Reshape_1Reshape max_pooling2d_1/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’,0:& "
 
_user_specified_nameinputs
­
±
0__inference_time_distributed_layer_call_fn_34203

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*5
_output_shapes#
!:’’’’’’’’’ė *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_318692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:’’’’’’’’’ė 2

Identity"
identityIdentity:output:0*<
_input_shapes+
):’’’’’’’’’ī::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Č
±
0__inference_time_distributed_layer_call_fn_34166

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*>
_output_shapes,
*:(’’’’’’’’’’’’’’’’’’ė *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_time_distributed_layer_call_and_return_conditional_losses_297532
StatefulPartitionedCall„
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*>
_output_shapes,
*:(’’’’’’’’’’’’’’’’’’ė 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:(’’’’’’’’’’’’’’’’’’ī::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ż
į
K__inference_time_distributed_layer_call_and_return_conditional_losses_29726

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2
identity¢conv2d/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ī        2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ī2	
Reshape¶
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:’’’’’’’’’ė *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_296542 
conv2d/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ė2
Reshape_1/shape/2i
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape­
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*>
_output_shapes,
*:(’’’’’’’’’’’’’’’’’’ė 2
	Reshape_1
IdentityIdentityReshape_1:output:0^conv2d/StatefulPartitionedCall*
T0*>
_output_shapes,
*:(’’’’’’’’’’’’’’’’’’ė 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:(’’’’’’’’’’’’’’’’’’ī::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ą
i
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_30274

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
ReshapeČ
flatten/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_302052
flatten/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :	2
Reshape_1/shape/2Ø
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshape flatten/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2
	Reshape_1t
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’	:& "
 
_user_specified_nameinputs
@
ą
'__forward_cudnn_gru_with_fallback_29618

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *<
api_implements*(gru_4417fa00-1c49-4e6a-ba14-dc50117fc1d4*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_cudnn_gru_with_fallback_29483_296192
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Ż
į
K__inference_time_distributed_layer_call_and_return_conditional_losses_29753

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2
identity¢conv2d/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ī        2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ī2	
Reshape¶
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:’’’’’’’’’ė *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_296542 
conv2d/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ė2
Reshape_1/shape/2i
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape­
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*>
_output_shapes,
*:(’’’’’’’’’’’’’’’’’’ė 2
	Reshape_1
IdentityIdentityReshape_1:output:0^conv2d/StatefulPartitionedCall*
T0*>
_output_shapes,
*:(’’’’’’’’’’’’’’’’’’ė 2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:(’’’’’’’’’’’’’’’’’’ī::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ė
i
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_34622

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
flatten/Const
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2
flatten/Reshapew
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’     2
Reshape_1/shape
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2
	Reshape_1k
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*2
_input_shapes!
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
Č
Ź
#__inference_gru_layer_call_fn_35436
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_318452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:’’’’’’’’’’’’’’’’’’	:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
Č
I
-__inference_max_pooling2d_layer_call_fn_29770

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_297642
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:& "
 
_user_specified_nameinputs
“
`
B__inference_dropout_layer_call_and_return_conditional_losses_30103

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:’’’’’’’’’	2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:’’’’’’’’’	:& "
 
_user_specified_nameinputs
Č
č

 __inference__wrapped_model_29641
input_1E
Asequential_time_distributed_conv2d_conv2d_readvariableop_resourceF
Bsequential_time_distributed_conv2d_biasadd_readvariableop_resourceI
Esequential_time_distributed_2_conv2d_1_conv2d_readvariableop_resourceJ
Fsequential_time_distributed_2_conv2d_1_biasadd_readvariableop_resource1
-sequential_gru_statefulpartitionedcall_args_21
-sequential_gru_statefulpartitionedcall_args_31
-sequential_gru_statefulpartitionedcall_args_43
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource5
1sequential_dense_2_matmul_readvariableop_resource6
2sequential_dense_2_biasadd_readvariableop_resource
identity¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢)sequential/dense_2/BiasAdd/ReadVariableOp¢(sequential/dense_2/MatMul/ReadVariableOp¢&sequential/gru/StatefulPartitionedCall¢9sequential/time_distributed/conv2d/BiasAdd/ReadVariableOp¢8sequential/time_distributed/conv2d/Conv2D/ReadVariableOp¢=sequential/time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp¢<sequential/time_distributed_2/conv2d_1/Conv2D/ReadVariableOpÆ
)sequential/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ī        2+
)sequential/time_distributed/Reshape/shapeĪ
#sequential/time_distributed/ReshapeReshapeinput_12sequential/time_distributed/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ī2%
#sequential/time_distributed/Reshapež
8sequential/time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOpAsequential_time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02:
8sequential/time_distributed/conv2d/Conv2D/ReadVariableOpµ
)sequential/time_distributed/conv2d/Conv2DConv2D,sequential/time_distributed/Reshape:output:0@sequential/time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė *
paddingVALID*
strides
2+
)sequential/time_distributed/conv2d/Conv2Dõ
9sequential/time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOpBsequential_time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9sequential/time_distributed/conv2d/BiasAdd/ReadVariableOp
*sequential/time_distributed/conv2d/BiasAddBiasAdd2sequential/time_distributed/conv2d/Conv2D:output:0Asequential/time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2,
*sequential/time_distributed/conv2d/BiasAddĖ
'sequential/time_distributed/conv2d/ReluRelu3sequential/time_distributed/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2)
'sequential/time_distributed/conv2d/Relu·
+sequential/time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ė         2-
+sequential/time_distributed/Reshape_1/shape
%sequential/time_distributed/Reshape_1Reshape5sequential/time_distributed/conv2d/Relu:activations:04sequential/time_distributed/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’ė 2'
%sequential/time_distributed/Reshape_1³
+sequential/time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’ė         2-
+sequential/time_distributed_1/Reshape/shapeū
%sequential/time_distributed_1/ReshapeReshape.sequential/time_distributed/Reshape_1:output:04sequential/time_distributed_1/Reshape/shape:output:0*
T0*1
_output_shapes
:’’’’’’’’’ė 2'
%sequential/time_distributed_1/Reshape
3sequential/time_distributed_1/max_pooling2d/MaxPoolMaxPool.sequential/time_distributed_1/Reshape:output:0*/
_output_shapes
:’’’’’’’’’/3 *
ksize
*
paddingVALID*
strides
25
3sequential/time_distributed_1/max_pooling2d/MaxPool»
-sequential/time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   /   3       2/
-sequential/time_distributed_1/Reshape_1/shape
'sequential/time_distributed_1/Reshape_1Reshape<sequential/time_distributed_1/max_pooling2d/MaxPool:output:06sequential/time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’/3 2)
'sequential/time_distributed_1/Reshape_1³
+sequential/time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’/   3       2-
+sequential/time_distributed_2/Reshape/shapeū
%sequential/time_distributed_2/ReshapeReshape0sequential/time_distributed_1/Reshape_1:output:04sequential/time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’/3 2'
%sequential/time_distributed_2/Reshape
<sequential/time_distributed_2/conv2d_1/Conv2D/ReadVariableOpReadVariableOpEsequential_time_distributed_2_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02>
<sequential/time_distributed_2/conv2d_1/Conv2D/ReadVariableOpĮ
-sequential/time_distributed_2/conv2d_1/Conv2DConv2D.sequential/time_distributed_2/Reshape:output:0Dsequential/time_distributed_2/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,0*
paddingVALID*
strides
2/
-sequential/time_distributed_2/conv2d_1/Conv2D
=sequential/time_distributed_2/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpFsequential_time_distributed_2_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=sequential/time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp¤
.sequential/time_distributed_2/conv2d_1/BiasAddBiasAdd6sequential/time_distributed_2/conv2d_1/Conv2D:output:0Esequential/time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’,020
.sequential/time_distributed_2/conv2d_1/BiasAddÕ
+sequential/time_distributed_2/conv2d_1/ReluRelu7sequential/time_distributed_2/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02-
+sequential/time_distributed_2/conv2d_1/Relu»
-sequential/time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’   ,   0      2/
-sequential/time_distributed_2/Reshape_1/shape
'sequential/time_distributed_2/Reshape_1Reshape9sequential/time_distributed_2/conv2d_1/Relu:activations:06sequential/time_distributed_2/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’,02)
'sequential/time_distributed_2/Reshape_1³
+sequential/time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’,   0      2-
+sequential/time_distributed_3/Reshape/shapeū
%sequential/time_distributed_3/ReshapeReshape0sequential/time_distributed_2/Reshape_1:output:04sequential/time_distributed_3/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02'
%sequential/time_distributed_3/Reshape
5sequential/time_distributed_3/max_pooling2d_1/MaxPoolMaxPool.sequential/time_distributed_3/Reshape:output:0*/
_output_shapes
:’’’’’’’’’	*
ksize
*
paddingVALID*
strides
27
5sequential/time_distributed_3/max_pooling2d_1/MaxPool»
-sequential/time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2/
-sequential/time_distributed_3/Reshape_1/shape
'sequential/time_distributed_3/Reshape_1Reshape>sequential/time_distributed_3/max_pooling2d_1/MaxPool:output:06sequential/time_distributed_3/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2)
'sequential/time_distributed_3/Reshape_1³
+sequential/time_distributed_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2-
+sequential/time_distributed_4/Reshape/shapeū
%sequential/time_distributed_4/ReshapeReshape0sequential/time_distributed_3/Reshape_1:output:04sequential/time_distributed_4/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2'
%sequential/time_distributed_4/ReshapeÖ
.sequential/time_distributed_4/dropout/IdentityIdentity.sequential/time_distributed_4/Reshape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	20
.sequential/time_distributed_4/dropout/Identity»
-sequential/time_distributed_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"’’’’      	      2/
-sequential/time_distributed_4/Reshape_1/shape
'sequential/time_distributed_4/Reshape_1Reshape7sequential/time_distributed_4/dropout/Identity:output:06sequential/time_distributed_4/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:’’’’’’’’’	2)
'sequential/time_distributed_4/Reshape_1³
+sequential/time_distributed_5/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2-
+sequential/time_distributed_5/Reshape/shapeū
%sequential/time_distributed_5/ReshapeReshape0sequential/time_distributed_4/Reshape_1:output:04sequential/time_distributed_5/Reshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2'
%sequential/time_distributed_5/Reshape«
+sequential/time_distributed_5/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2-
+sequential/time_distributed_5/flatten/Const
-sequential/time_distributed_5/flatten/ReshapeReshape.sequential/time_distributed_5/Reshape:output:04sequential/time_distributed_5/flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’	2/
-sequential/time_distributed_5/flatten/Reshape³
-sequential/time_distributed_5/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"’’’’     2/
-sequential/time_distributed_5/Reshape_1/shape
'sequential/time_distributed_5/Reshape_1Reshape6sequential/time_distributed_5/flatten/Reshape:output:06sequential/time_distributed_5/Reshape_1/shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’	2)
'sequential/time_distributed_5/Reshape_1
sequential/gru/ShapeShape0sequential/time_distributed_5/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential/gru/Shape
"sequential/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/gru/strided_slice/stack
$sequential/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/gru/strided_slice/stack_1
$sequential/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/gru/strided_slice/stack_2¼
sequential/gru/strided_sliceStridedSlicesequential/gru/Shape:output:0+sequential/gru/strided_slice/stack:output:0-sequential/gru/strided_slice/stack_1:output:0-sequential/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sequential/gru/strided_slice{
sequential/gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
sequential/gru/zeros/mul/yØ
sequential/gru/zeros/mulMul%sequential/gru/strided_slice:output:0#sequential/gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/zeros/mul}
sequential/gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
sequential/gru/zeros/Less/y£
sequential/gru/zeros/LessLesssequential/gru/zeros/mul:z:0$sequential/gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/zeros/Less
sequential/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
sequential/gru/zeros/packed/1æ
sequential/gru/zeros/packedPack%sequential/gru/strided_slice:output:0&sequential/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/gru/zeros/packed}
sequential/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/gru/zeros/Const²
sequential/gru/zerosFill$sequential/gru/zeros/packed:output:0#sequential/gru/zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential/gru/zerosĖ
&sequential/gru/StatefulPartitionedCallStatefulPartitionedCall0sequential/time_distributed_5/Reshape_1:output:0sequential/gru/zeros:output:0-sequential_gru_statefulpartitionedcall_args_2-sequential_gru_statefulpartitionedcall_args_3-sequential_gru_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*V
_output_shapesD
B:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference_standard_gru_293972(
&sequential/gru/StatefulPartitionedCall®
sequential/dropout_1/IdentityIdentity/sequential/gru/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential/dropout_1/IdentityĮ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	<*
dtype02(
&sequential/dense/MatMul/ReadVariableOpĘ
sequential/dense/MatMulMatMul&sequential/dropout_1/Identity:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
sequential/dense/MatMulæ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÅ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’<2
sequential/dense/BiasAddĘ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:<(*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpĒ
sequential/dense_1/MatMulMatMul!sequential/dense/BiasAdd:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’(2
sequential/dense_1/MatMulÅ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpĶ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’(2
sequential/dense_1/BiasAddĘ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOpÉ
sequential/dense_2/MatMulMatMul#sequential/dense_1/BiasAdd:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential/dense_2/MatMulÅ
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOpĶ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential/dense_2/BiasAdd
sequential/dense_2/SigmoidSigmoid#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
sequential/dense_2/Sigmoid
IdentityIdentitysequential/dense_2/Sigmoid:y:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp'^sequential/gru/StatefulPartitionedCall:^sequential/time_distributed/conv2d/BiasAdd/ReadVariableOp9^sequential/time_distributed/conv2d/Conv2D/ReadVariableOp>^sequential/time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp=^sequential/time_distributed_2/conv2d_1/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:’’’’’’’’’ī:::::::::::::2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2P
&sequential/gru/StatefulPartitionedCall&sequential/gru/StatefulPartitionedCall2v
9sequential/time_distributed/conv2d/BiasAdd/ReadVariableOp9sequential/time_distributed/conv2d/BiasAdd/ReadVariableOp2t
8sequential/time_distributed/conv2d/Conv2D/ReadVariableOp8sequential/time_distributed/conv2d/Conv2D/ReadVariableOp2~
=sequential/time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp=sequential/time_distributed_2/conv2d_1/BiasAdd/ReadVariableOp2|
<sequential/time_distributed_2/conv2d_1/Conv2D/ReadVariableOp<sequential/time_distributed_2/conv2d_1/Conv2D/ReadVariableOp:' #
!
_user_specified_name	input_1
 
i
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_30064

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’,   0      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’,02	
Reshapeē
max_pooling2d_1/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_299772!
max_pooling2d_1/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4ą
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape¬
	Reshape_1Reshape(max_pooling2d_1/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’,0:& "
 
_user_specified_nameinputs
ą
i
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_30253

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"’’’’   	      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2	
ReshapeČ
flatten/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_302052
flatten/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2
Reshape_1/shape/0i
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :	2
Reshape_1/shape/2Ø
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape
	Reshape_1Reshape flatten/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2
	Reshape_1t
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’	:& "
 
_user_specified_nameinputs
Ø
N
2__inference_time_distributed_3_layer_call_fn_34436

inputs
identityĶ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_300642
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:&’’’’’’’’’’’’’’’’’’,0:& "
 
_user_specified_nameinputs
@
ą
'__forward_cudnn_gru_with_fallback_35028

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dim¢CudnnRNN¢Reshape/ReadVariableOp¢split/ReadVariableOp¢split_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDims8
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1U
Reshape/ReadVariableOpReadVariableOpbias*
dtype02
Reshape/ReadVariableOpU
Reshape/shapeConst*
dtype0*
valueB:
’’’’’’’’’2
Reshape/shape^
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T02	
Reshape<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimc
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
’’’’’’’’’2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T02
transpose_1M
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm]
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_6N
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_12D
concat/axisConst*
dtype0*
value	B : 2
concat/axis¹
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concatQ
CudnnRNN/input_cConst*
dtype0*
valueB
 *    2
CudnnRNN/input_c
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegru2

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
’’’’’’’’’2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2Ķ
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_7/permConst*
dtype0*!
valueB"          2
transpose_7/perm^
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T02
transpose_7R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeN
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtime
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1

Identity_2IdentitySqueeze:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *<
api_implements*(gru_214693d3-9c8e-4a5e-a5f4-04ba364da9ac*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_cudnn_gru_with_fallback_34893_350292
CudnnRNNCudnnRNN20
Reshape/ReadVariableOpReshape/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Ę
³
2__inference_time_distributed_2_layer_call_fn_34346

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’,0*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_299662
StatefulPartitionedCall£
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*<
_output_shapes*
(:&’’’’’’’’’’’’’’’’’’,02

Identity"
identityIdentity:output:0*C
_input_shapes2
0:&’’’’’’’’’’’’’’’’’’/3 ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ų
ć
>__inference_gru_layer_call_and_return_conditional_losses_31845

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ā
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
zerosŠ
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*_
_output_shapesM
K:’’’’’’’’’:’’’’’’’’’’’’’’’’’’:’’’’’’’’’: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference_standard_gru_316212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:’’’’’’’’’’’’’’’’’’	:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
å
b
)__inference_dropout_1_layer_call_fn_36260

inputs
identity¢StatefulPartitionedCallČ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:’’’’’’’’’*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_329002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
¦
ź
:__inference___backward_cudnn_gru_with_fallback_33910_34046
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4¢(gradients/CudnnRNN_grad/CudnnRNNBackpropv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_0|
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:’’’’’’’’’2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3£
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeę
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*,
_output_shapes
:’’’’’’’’’*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGradĢ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationą
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’2&
$gradients/transpose_7_grad/transpose
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/ShapeĒ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:’’’’’’’’’2 
gradients/Squeeze_grad/Reshape
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*,
_output_shapes
:’’’’’’’’’2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_like
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::’’’’’’’’’	:’’’’’’’’’: :*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackpropÄ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation÷
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:’’’’’’’’’	2$
"gradients/transpose_grad/transpose
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeė
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/RankÆ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/mod
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_1
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:	2
gradients/concat_grad/Shape_2
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_3
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_4
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_5
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_6
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_7
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_8
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:2
gradients/concat_grad/Shape_9
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_10
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:2 
gradients/concat_grad/Shape_11¾
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffset
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_1
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes

:	2
gradients/concat_grad/Slice_2
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_3
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_4
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:2
gradients/concat_grad/Slice_5
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_6
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_7
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_8
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:2
gradients/concat_grad/Slice_9
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_10
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:2 
gradients/concat_grad/Slice_11
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_1_grad/ShapeÉ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_1_grad/Reshape
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_2_grad/ShapeĖ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_2_grad/Reshape
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2 
gradients/Reshape_3_grad/ShapeĖ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0* 
_output_shapes
:
	2"
 gradients/Reshape_3_grad/Reshape
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_4_grad/ShapeĖ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_4_grad/Reshape
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_5_grad/ShapeĖ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_5_grad/Reshape
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
gradients/Reshape_6_grad/ShapeĖ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
2"
 gradients/Reshape_6_grad/Reshape
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_7_grad/ShapeĘ
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_7_grad/Reshape
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_8_grad/ShapeĘ
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_8_grad/Reshape
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2 
gradients/Reshape_9_grad/ShapeĘ
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:2"
 gradients/Reshape_9_grad/Reshape
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_10_grad/ShapeÉ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_10_grad/Reshape
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_11_grad/ShapeŹ
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_11_grad/Reshape
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2!
gradients/Reshape_12_grad/ShapeŹ
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:2#
!gradients/Reshape_12_grad/ReshapeĢ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationį
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_1_grad/transposeĢ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationį
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_2_grad/transposeĢ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationį
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
	2&
$gradients/transpose_3_grad/transposeĢ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationį
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_4_grad/transposeĢ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationį
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_5_grad/transposeĢ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationį
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
2&
$gradients/transpose_6_grad/transposeÆ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:2
gradients/split_2_grad/concat„
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0* 
_output_shapes
:
	2
gradients/split_grad/concat­
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
2
gradients/split_1_grad/concat
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
gradients/Reshape_grad/ShapeÄ
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	2 
gradients/Reshape_grad/ReshapeŖ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:’’’’’’’’’	2

Identity®

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*(
_output_shapes
:’’’’’’’’’2

Identity_1 

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
	2

Identity_2¢

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0* 
_output_shapes
:
2

Identity_3¢

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*„
_input_shapes
:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: :’’’’’’’’’:::::’’’’’’’’’: ::’’’’’’’’’	:’’’’’’’’’: :::’’’’’’’’’: ::::::: : : *<
api_implements*(gru_610e0db3-a6a3-4ac2-b3aa-b93ab2fb5213*
api_preferred_deviceGPU*B
forward_function_name)'__forward_cudnn_gru_with_fallback_340452T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
ē
a
B__inference_dropout_layer_call_and_return_conditional_losses_36337

inputs
identitya
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/random_uniform/max¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:’’’’’’’’’	*
dtype02&
$dropout/random_uniform/RandomUniformŖ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/subČ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/random_uniform/mul¶
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv©
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/mul
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:’’’’’’’’’	2
dropout/Cast
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:’’’’’’’’’	2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:’’’’’’’’’	2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’	:& "
 
_user_specified_nameinputs"ÆL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ø
serving_default¤
I
input_1>
serving_default_input_1:0’’’’’’’’’ī;
dense_20
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:„
ńV
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
ą_default_save_signature
į__call__
+ā&call_and_return_all_conditional_losses"éR
_tf_keras_sequentialŹR{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "TimeDistributed", "config": {"name": "time_distributed", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "batch_input_shape": [null, 3, 238, 260, 3]}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [5, 5], "padding": "valid", "strides": [5, 5], "data_format": "channels_last"}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_3", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [5, 5], "padding": "valid", "strides": [5, 5], "data_format": "channels_last"}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_4", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_5", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}}}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.001, "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 60, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 238, 260, 3], "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "TimeDistributed", "config": {"name": "time_distributed", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "batch_input_shape": [null, 3, 238, 260, 3]}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [5, 5], "padding": "valid", "strides": [5, 5], "data_format": "channels_last"}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_3", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [5, 5], "padding": "valid", "strides": [5, 5], "data_format": "channels_last"}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_4", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}}}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_5", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}}}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.001, "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 60, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
·"“
_tf_keras_input_layer{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 3, 238, 260, 3], "config": {"batch_input_shape": [null, 3, 238, 260, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
	
	layer

_input_map
regularization_losses
trainable_variables
	variables
	keras_api
ć__call__
+ä&call_and_return_all_conditional_losses"Õ
_tf_keras_layer»{"class_name": "TimeDistributed", "name": "time_distributed", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "time_distributed", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 238, 260, 3], "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
­
	layer

_input_map
regularization_losses
trainable_variables
	variables
	keras_api
å__call__
+ę&call_and_return_all_conditional_losses"
_tf_keras_layerē{"class_name": "TimeDistributed", "name": "time_distributed_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [5, 5], "padding": "valid", "strides": [5, 5], "data_format": "channels_last"}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 235, 257, 32], "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
	
	layer
 
_input_map
!regularization_losses
"trainable_variables
#	variables
$	keras_api
ē__call__
+č&call_and_return_all_conditional_losses"Ś
_tf_keras_layerĄ{"class_name": "TimeDistributed", "name": "time_distributed_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "time_distributed_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 47, 51, 32], "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
­
	%layer
&
_input_map
'regularization_losses
(trainable_variables
)	variables
*	keras_api
é__call__
+ź&call_and_return_all_conditional_losses"
_tf_keras_layerē{"class_name": "TimeDistributed", "name": "time_distributed_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "time_distributed_3", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [5, 5], "padding": "valid", "strides": [5, 5], "data_format": "channels_last"}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 44, 48, 16], "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ó
	+layer
,
_input_map
-regularization_losses
.trainable_variables
/	variables
0	keras_api
ė__call__
+ģ&call_and_return_all_conditional_losses"Ē
_tf_keras_layer­{"class_name": "TimeDistributed", "name": "time_distributed_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "time_distributed_4", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 8, 9, 16], "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ā
	1layer
2
_input_map
3regularization_losses
4trainable_variables
5	variables
6	keras_api
ķ__call__
+ī&call_and_return_all_conditional_losses"¶
_tf_keras_layer{"class_name": "TimeDistributed", "name": "time_distributed_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "time_distributed_5", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 8, 9, 16], "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¦

7cell
8
state_spec
9regularization_losses
:trainable_variables
;	variables
<	keras_api
ļ__call__
+š&call_and_return_all_conditional_losses"ū
_tf_keras_layerį{"class_name": "GRU", "name": "gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.001, "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 1152], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
²
=regularization_losses
>trainable_variables
?	variables
@	keras_api
ń__call__
+ņ&call_and_return_all_conditional_losses"”
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
ņ

Akernel
Bbias
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
ó__call__
+ō&call_and_return_all_conditional_losses"Ė
_tf_keras_layer±{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 60, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
õ

Gkernel
Hbias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"Ī
_tf_keras_layer“{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}}
õ

Mkernel
Nbias
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
÷__call__
+ų&call_and_return_all_conditional_losses"Ī
_tf_keras_layer“{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}}
ī
Siter
	Tdecay
Ulearning_rate
Vmomentum
Wrho
ArmsÓ
BrmsŌ
GrmsÕ
HrmsÖ
Mrms×
NrmsŲ
XrmsŁ
YrmsŚ
ZrmsŪ
[rmsÜ
\rmsŻ
]rmsŽ
^rmsß"
	optimizer
 "
trackable_list_wrapper
~
X0
Y1
Z2
[3
\4
]5
^6
A7
B8
G9
H10
M11
N12"
trackable_list_wrapper
~
X0
Y1
Z2
[3
\4
]5
^6
A7
B8
G9
H10
M11
N12"
trackable_list_wrapper
»
regularization_losses

_layers
trainable_variables
`layer_regularization_losses
anon_trainable_variables
bmetrics
	variables
į__call__
ą_default_save_signature
+ā&call_and_return_all_conditional_losses
'ā"call_and_return_conditional_losses"
_generic_user_object
-
łserving_default"
signature_map
ź

Xkernel
Ybias
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
ś__call__
+ū&call_and_return_all_conditional_losses"Ć
_tf_keras_layer©{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper

regularization_losses
trainable_variables
glayer_regularization_losses
hnon_trainable_variables
imetrics
	variables

jlayers
ć__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
ū
kregularization_losses
ltrainable_variables
m	variables
n	keras_api
ü__call__
+ż&call_and_return_all_conditional_losses"ź
_tf_keras_layerŠ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [5, 5], "padding": "valid", "strides": [5, 5], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

regularization_losses
trainable_variables
olayer_regularization_losses
pnon_trainable_variables
qmetrics
	variables

rlayers
å__call__
+ę&call_and_return_all_conditional_losses
'ę"call_and_return_conditional_losses"
_generic_user_object
ļ

Zkernel
[bias
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
ž__call__
+’&call_and_return_all_conditional_losses"Č
_tf_keras_layer®{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [4, 4], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper

!regularization_losses
"trainable_variables
wlayer_regularization_losses
xnon_trainable_variables
ymetrics
#	variables

zlayers
ē__call__
+č&call_and_return_all_conditional_losses
'č"call_and_return_conditional_losses"
_generic_user_object
’
{regularization_losses
|trainable_variables
}	variables
~	keras_api
__call__
+&call_and_return_all_conditional_losses"ī
_tf_keras_layerŌ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [5, 5], "padding": "valid", "strides": [5, 5], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 
'regularization_losses
(trainable_variables
layer_regularization_losses
non_trainable_variables
metrics
)	variables
layers
é__call__
+ź&call_and_return_all_conditional_losses
'ź"call_and_return_conditional_losses"
_generic_user_object
²
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
”
-regularization_losses
.trainable_variables
 layer_regularization_losses
non_trainable_variables
metrics
/	variables
layers
ė__call__
+ģ&call_and_return_all_conditional_losses
'ģ"call_and_return_conditional_losses"
_generic_user_object
²
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
”
3regularization_losses
4trainable_variables
 layer_regularization_losses
non_trainable_variables
metrics
5	variables
layers
ķ__call__
+ī&call_and_return_all_conditional_losses
'ī"call_and_return_conditional_losses"
_generic_user_object


\kernel
]recurrent_kernel
^bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"É
_tf_keras_layerÆ{"class_name": "GRUCell", "name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.001, "seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
\0
]1
^2"
trackable_list_wrapper
5
\0
]1
^2"
trackable_list_wrapper
”
9regularization_losses
:trainable_variables
 layer_regularization_losses
non_trainable_variables
metrics
;	variables
layers
ļ__call__
+š&call_and_return_all_conditional_losses
'š"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
”
=regularization_losses
>trainable_variables
 layer_regularization_losses
non_trainable_variables
metrics
?	variables
layers
ń__call__
+ņ&call_and_return_all_conditional_losses
'ņ"call_and_return_conditional_losses"
_generic_user_object
:	<2dense/kernel
:<2
dense/bias
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
”
Cregularization_losses
Dtrainable_variables
 layer_regularization_losses
 non_trainable_variables
”metrics
E	variables
¢layers
ó__call__
+ō&call_and_return_all_conditional_losses
'ō"call_and_return_conditional_losses"
_generic_user_object
 :<(2dense_1/kernel
:(2dense_1/bias
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
”
Iregularization_losses
Jtrainable_variables
 £layer_regularization_losses
¤non_trainable_variables
„metrics
K	variables
¦layers
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
 :(2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
”
Oregularization_losses
Ptrainable_variables
 §layer_regularization_losses
Ønon_trainable_variables
©metrics
Q	variables
Ŗlayers
÷__call__
+ų&call_and_return_all_conditional_losses
'ų"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
1:/ 2time_distributed/kernel
#:! 2time_distributed/bias
3:1 2time_distributed_2/kernel
%:#2time_distributed_2/bias
:
	2
gru/kernel
(:&
2gru/recurrent_kernel
:	2gru/bias
n
0
1
2
3
4
5
6
	7

8
9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
«0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
”
cregularization_losses
dtrainable_variables
 ¬layer_regularization_losses
­non_trainable_variables
®metrics
e	variables
Ælayers
ś__call__
+ū&call_and_return_all_conditional_losses
'ū"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
”
kregularization_losses
ltrainable_variables
 °layer_regularization_losses
±non_trainable_variables
²metrics
m	variables
³layers
ü__call__
+ż&call_and_return_all_conditional_losses
'ż"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
”
sregularization_losses
ttrainable_variables
 “layer_regularization_losses
µnon_trainable_variables
¶metrics
u	variables
·layers
ž__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
”
{regularization_losses
|trainable_variables
 ølayer_regularization_losses
¹non_trainable_variables
ŗmetrics
}	variables
»layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
%0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
regularization_losses
trainable_variables
 ¼layer_regularization_losses
½non_trainable_variables
¾metrics
	variables
ælayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¤
regularization_losses
trainable_variables
 Ąlayer_regularization_losses
Įnon_trainable_variables
Āmetrics
	variables
Ćlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
 "
trackable_list_wrapper
5
\0
]1
^2"
trackable_list_wrapper
5
\0
]1
^2"
trackable_list_wrapper
¤
regularization_losses
trainable_variables
 Älayer_regularization_losses
Ånon_trainable_variables
Ęmetrics
	variables
Ēlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
£

Čtotal

Écount
Ź
_fn_kwargs
Ėregularization_losses
Ģtrainable_variables
Ķ	variables
Ī	keras_api
__call__
+&call_and_return_all_conditional_losses"å
_tf_keras_layerĖ{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Č0
É1"
trackable_list_wrapper
¤
Ėregularization_losses
Ģtrainable_variables
 Ļlayer_regularization_losses
Šnon_trainable_variables
Ńmetrics
Ķ	variables
Ņlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
Č0
É1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
):'	<2RMSprop/dense/kernel/rms
": <2RMSprop/dense/bias/rms
*:(<(2RMSprop/dense_1/kernel/rms
$:"(2RMSprop/dense_1/bias/rms
*:((2RMSprop/dense_2/kernel/rms
$:"2RMSprop/dense_2/bias/rms
;:9 2#RMSprop/time_distributed/kernel/rms
-:+ 2!RMSprop/time_distributed/bias/rms
=:; 2%RMSprop/time_distributed_2/kernel/rms
/:-2#RMSprop/time_distributed_2/bias/rms
(:&
	2RMSprop/gru/kernel/rms
2:0
2 RMSprop/gru/recurrent_kernel/rms
%:#	2RMSprop/gru/bias/rms
ģ2é
 __inference__wrapped_model_29641Ä
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *4¢1
/,
input_1’’’’’’’’’ī
ö2ó
*__inference_sequential_layer_call_fn_33107
*__inference_sequential_layer_call_fn_34104
*__inference_sequential_layer_call_fn_34086
*__inference_sequential_layer_call_fn_33061Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ā2ß
E__inference_sequential_layer_call_and_return_conditional_losses_32986
E__inference_sequential_layer_call_and_return_conditional_losses_33616
E__inference_sequential_layer_call_and_return_conditional_losses_34068
E__inference_sequential_layer_call_and_return_conditional_losses_33014Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
0__inference_time_distributed_layer_call_fn_34210
0__inference_time_distributed_layer_call_fn_34166
0__inference_time_distributed_layer_call_fn_34203
0__inference_time_distributed_layer_call_fn_34159Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ś2÷
K__inference_time_distributed_layer_call_and_return_conditional_losses_34128
K__inference_time_distributed_layer_call_and_return_conditional_losses_34152
K__inference_time_distributed_layer_call_and_return_conditional_losses_34181
K__inference_time_distributed_layer_call_and_return_conditional_losses_34196Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
2__inference_time_distributed_1_layer_call_fn_34233
2__inference_time_distributed_1_layer_call_fn_34238
2__inference_time_distributed_1_layer_call_fn_34279
2__inference_time_distributed_1_layer_call_fn_34284Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2’
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34228
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34274
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34256
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34219Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
2__inference_time_distributed_2_layer_call_fn_34339
2__inference_time_distributed_2_layer_call_fn_34346
2__inference_time_distributed_2_layer_call_fn_34383
2__inference_time_distributed_2_layer_call_fn_34390Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2’
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_34308
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_34332
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_34376
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_34361Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
2__inference_time_distributed_3_layer_call_fn_34464
2__inference_time_distributed_3_layer_call_fn_34459
2__inference_time_distributed_3_layer_call_fn_34431
2__inference_time_distributed_3_layer_call_fn_34436Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2’
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_34445
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_34408
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_34426
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_34454Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
2__inference_time_distributed_4_layer_call_fn_34563
2__inference_time_distributed_4_layer_call_fn_34507
2__inference_time_distributed_4_layer_call_fn_34568
2__inference_time_distributed_4_layer_call_fn_34502Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2’
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_34558
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_34540
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_34488
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_34497Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
2__inference_time_distributed_5_layer_call_fn_34612
2__inference_time_distributed_5_layer_call_fn_34637
2__inference_time_distributed_5_layer_call_fn_34607
2__inference_time_distributed_5_layer_call_fn_34642Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2’
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_34602
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_34622
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_34585
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_34632Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ļ2ģ
#__inference_gru_layer_call_fn_35428
#__inference_gru_layer_call_fn_35436
#__inference_gru_layer_call_fn_36230
#__inference_gru_layer_call_fn_36222Õ
Ģ²Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ū2Ų
>__inference_gru_layer_call_and_return_conditional_losses_35031
>__inference_gru_layer_call_and_return_conditional_losses_35825
>__inference_gru_layer_call_and_return_conditional_losses_36214
>__inference_gru_layer_call_and_return_conditional_losses_35420Õ
Ģ²Č
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
)__inference_dropout_1_layer_call_fn_36260
)__inference_dropout_1_layer_call_fn_36265“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ę2Ć
D__inference_dropout_1_layer_call_and_return_conditional_losses_36255
D__inference_dropout_1_layer_call_and_return_conditional_losses_36250“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ļ2Ģ
%__inference_dense_layer_call_fn_36282¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ź2ē
@__inference_dense_layer_call_and_return_conditional_losses_36275¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ń2Ī
'__inference_dense_1_layer_call_fn_36299¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ģ2é
B__inference_dense_1_layer_call_and_return_conditional_losses_36292¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ń2Ī
'__inference_dense_2_layer_call_fn_36317¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ģ2é
B__inference_dense_2_layer_call_and_return_conditional_losses_36310¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
2B0
#__inference_signature_wrapper_33134input_1
2
&__inference_conv2d_layer_call_fn_29662×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *7¢4
2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’
 2
A__inference_conv2d_layer_call_and_return_conditional_losses_29654×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *7¢4
2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
-__inference_max_pooling2d_layer_call_fn_29770ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
°2­
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_29764ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
(__inference_conv2d_1_layer_call_fn_29875×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *7¢4
2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
¢2
C__inference_conv2d_1_layer_call_and_return_conditional_losses_29867×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *7¢4
2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
2
/__inference_max_pooling2d_1_layer_call_fn_29983ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
²2Æ
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_29977ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
'__inference_dropout_layer_call_fn_36347
'__inference_dropout_layer_call_fn_36352“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ā2æ
B__inference_dropout_layer_call_and_return_conditional_losses_36337
B__inference_dropout_layer_call_and_return_conditional_losses_36342“
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ń2Ī
'__inference_flatten_layer_call_fn_36363¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ģ2é
B__inference_flatten_layer_call_and_return_conditional_losses_36358¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ä2Į¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ä2Į¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ģ2ÉĘ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsŖ

trainingp 
annotationsŖ *
 
Ģ2ÉĘ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsŖ

trainingp 
annotationsŖ *
 §
 __inference__wrapped_model_29641XYZ[\]^ABGHMN>¢;
4¢1
/,
input_1’’’’’’’’’ī
Ŗ "1Ŗ.
,
dense_2!
dense_2’’’’’’’’’Ų
C__inference_conv2d_1_layer_call_and_return_conditional_losses_29867Z[I¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’
 °
(__inference_conv2d_1_layer_call_fn_29875Z[I¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’Ö
A__inference_conv2d_layer_call_and_return_conditional_losses_29654XYI¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "?¢<
52
0+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
 ®
&__inference_conv2d_layer_call_fn_29662XYI¢F
?¢<
:7
inputs+’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "2/+’’’’’’’’’’’’’’’’’’’’’’’’’’’ ¢
B__inference_dense_1_layer_call_and_return_conditional_losses_36292\GH/¢,
%¢"
 
inputs’’’’’’’’’<
Ŗ "%¢"

0’’’’’’’’’(
 z
'__inference_dense_1_layer_call_fn_36299OGH/¢,
%¢"
 
inputs’’’’’’’’’<
Ŗ "’’’’’’’’’(¢
B__inference_dense_2_layer_call_and_return_conditional_losses_36310\MN/¢,
%¢"
 
inputs’’’’’’’’’(
Ŗ "%¢"

0’’’’’’’’’
 z
'__inference_dense_2_layer_call_fn_36317OMN/¢,
%¢"
 
inputs’’’’’’’’’(
Ŗ "’’’’’’’’’”
@__inference_dense_layer_call_and_return_conditional_losses_36275]AB0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’<
 y
%__inference_dense_layer_call_fn_36282PAB0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’<¦
D__inference_dropout_1_layer_call_and_return_conditional_losses_36250^4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "&¢#

0’’’’’’’’’
 ¦
D__inference_dropout_1_layer_call_and_return_conditional_losses_36255^4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "&¢#

0’’’’’’’’’
 ~
)__inference_dropout_1_layer_call_fn_36260Q4¢1
*¢'
!
inputs’’’’’’’’’
p
Ŗ "’’’’’’’’’~
)__inference_dropout_1_layer_call_fn_36265Q4¢1
*¢'
!
inputs’’’’’’’’’
p 
Ŗ "’’’’’’’’’²
B__inference_dropout_layer_call_and_return_conditional_losses_36337l;¢8
1¢.
(%
inputs’’’’’’’’’	
p
Ŗ "-¢*
# 
0’’’’’’’’’	
 ²
B__inference_dropout_layer_call_and_return_conditional_losses_36342l;¢8
1¢.
(%
inputs’’’’’’’’’	
p 
Ŗ "-¢*
# 
0’’’’’’’’’	
 
'__inference_dropout_layer_call_fn_36347_;¢8
1¢.
(%
inputs’’’’’’’’’	
p
Ŗ " ’’’’’’’’’	
'__inference_dropout_layer_call_fn_36352_;¢8
1¢.
(%
inputs’’’’’’’’’	
p 
Ŗ " ’’’’’’’’’	§
B__inference_flatten_layer_call_and_return_conditional_losses_36358a7¢4
-¢*
(%
inputs’’’’’’’’’	
Ŗ "&¢#

0’’’’’’’’’	
 
'__inference_flatten_layer_call_fn_36363T7¢4
-¢*
(%
inputs’’’’’’’’’	
Ŗ "’’’’’’’’’	Į
>__inference_gru_layer_call_and_return_conditional_losses_35031\]^P¢M
F¢C
52
0-
inputs/0’’’’’’’’’’’’’’’’’’	

 
p

 
Ŗ "&¢#

0’’’’’’’’’
 Į
>__inference_gru_layer_call_and_return_conditional_losses_35420\]^P¢M
F¢C
52
0-
inputs/0’’’’’’’’’’’’’’’’’’	

 
p 

 
Ŗ "&¢#

0’’’’’’’’’
 ±
>__inference_gru_layer_call_and_return_conditional_losses_35825o\]^@¢=
6¢3
%"
inputs’’’’’’’’’	

 
p

 
Ŗ "&¢#

0’’’’’’’’’
 ±
>__inference_gru_layer_call_and_return_conditional_losses_36214o\]^@¢=
6¢3
%"
inputs’’’’’’’’’	

 
p 

 
Ŗ "&¢#

0’’’’’’’’’
 
#__inference_gru_layer_call_fn_35428r\]^P¢M
F¢C
52
0-
inputs/0’’’’’’’’’’’’’’’’’’	

 
p

 
Ŗ "’’’’’’’’’
#__inference_gru_layer_call_fn_35436r\]^P¢M
F¢C
52
0-
inputs/0’’’’’’’’’’’’’’’’’’	

 
p 

 
Ŗ "’’’’’’’’’
#__inference_gru_layer_call_fn_36222b\]^@¢=
6¢3
%"
inputs’’’’’’’’’	

 
p

 
Ŗ "’’’’’’’’’
#__inference_gru_layer_call_fn_36230b\]^@¢=
6¢3
%"
inputs’’’’’’’’’	

 
p 

 
Ŗ "’’’’’’’’’ķ
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_29977R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Å
/__inference_max_pooling2d_1_layer_call_fn_29983R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’ė
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_29764R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ć
-__inference_max_pooling2d_layer_call_fn_29770R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’Ē
E__inference_sequential_layer_call_and_return_conditional_losses_32986~XYZ[\]^ABGHMNF¢C
<¢9
/,
input_1’’’’’’’’’ī
p

 
Ŗ "%¢"

0’’’’’’’’’
 Ē
E__inference_sequential_layer_call_and_return_conditional_losses_33014~XYZ[\]^ABGHMNF¢C
<¢9
/,
input_1’’’’’’’’’ī
p 

 
Ŗ "%¢"

0’’’’’’’’’
 Ę
E__inference_sequential_layer_call_and_return_conditional_losses_33616}XYZ[\]^ABGHMNE¢B
;¢8
.+
inputs’’’’’’’’’ī
p

 
Ŗ "%¢"

0’’’’’’’’’
 Ę
E__inference_sequential_layer_call_and_return_conditional_losses_34068}XYZ[\]^ABGHMNE¢B
;¢8
.+
inputs’’’’’’’’’ī
p 

 
Ŗ "%¢"

0’’’’’’’’’
 
*__inference_sequential_layer_call_fn_33061qXYZ[\]^ABGHMNF¢C
<¢9
/,
input_1’’’’’’’’’ī
p

 
Ŗ "’’’’’’’’’
*__inference_sequential_layer_call_fn_33107qXYZ[\]^ABGHMNF¢C
<¢9
/,
input_1’’’’’’’’’ī
p 

 
Ŗ "’’’’’’’’’
*__inference_sequential_layer_call_fn_34086pXYZ[\]^ABGHMNE¢B
;¢8
.+
inputs’’’’’’’’’ī
p

 
Ŗ "’’’’’’’’’
*__inference_sequential_layer_call_fn_34104pXYZ[\]^ABGHMNE¢B
;¢8
.+
inputs’’’’’’’’’ī
p 

 
Ŗ "’’’’’’’’’µ
#__inference_signature_wrapper_33134XYZ[\]^ABGHMNI¢F
¢ 
?Ŗ<
:
input_1/,
input_1’’’’’’’’’ī"1Ŗ.
,
dense_2!
dense_2’’’’’’’’’Ė
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34219zE¢B
;¢8
.+
inputs’’’’’’’’’ė 
p

 
Ŗ "1¢.
'$
0’’’’’’’’’/3 
 Ė
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34228zE¢B
;¢8
.+
inputs’’’’’’’’’ė 
p 

 
Ŗ "1¢.
'$
0’’’’’’’’’/3 
 Ž
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34256N¢K
D¢A
74
inputs(’’’’’’’’’’’’’’’’’’ė 
p

 
Ŗ ":¢7
0-
0&’’’’’’’’’’’’’’’’’’/3 
 Ž
M__inference_time_distributed_1_layer_call_and_return_conditional_losses_34274N¢K
D¢A
74
inputs(’’’’’’’’’’’’’’’’’’ė 
p 

 
Ŗ ":¢7
0-
0&’’’’’’’’’’’’’’’’’’/3 
 £
2__inference_time_distributed_1_layer_call_fn_34233mE¢B
;¢8
.+
inputs’’’’’’’’’ė 
p

 
Ŗ "$!’’’’’’’’’/3 £
2__inference_time_distributed_1_layer_call_fn_34238mE¢B
;¢8
.+
inputs’’’’’’’’’ė 
p 

 
Ŗ "$!’’’’’’’’’/3 µ
2__inference_time_distributed_1_layer_call_fn_34279N¢K
D¢A
74
inputs(’’’’’’’’’’’’’’’’’’ė 
p

 
Ŗ "-*&’’’’’’’’’’’’’’’’’’/3 µ
2__inference_time_distributed_1_layer_call_fn_34284N¢K
D¢A
74
inputs(’’’’’’’’’’’’’’’’’’ė 
p 

 
Ŗ "-*&’’’’’’’’’’’’’’’’’’/3 ą
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_34308Z[L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’/3 
p

 
Ŗ ":¢7
0-
0&’’’’’’’’’’’’’’’’’’,0
 ą
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_34332Z[L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’/3 
p 

 
Ŗ ":¢7
0-
0&’’’’’’’’’’’’’’’’’’,0
 Ķ
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_34361|Z[C¢@
9¢6
,)
inputs’’’’’’’’’/3 
p

 
Ŗ "1¢.
'$
0’’’’’’’’’,0
 Ķ
M__inference_time_distributed_2_layer_call_and_return_conditional_losses_34376|Z[C¢@
9¢6
,)
inputs’’’’’’’’’/3 
p 

 
Ŗ "1¢.
'$
0’’’’’’’’’,0
 ø
2__inference_time_distributed_2_layer_call_fn_34339Z[L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’/3 
p

 
Ŗ "-*&’’’’’’’’’’’’’’’’’’,0ø
2__inference_time_distributed_2_layer_call_fn_34346Z[L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’/3 
p 

 
Ŗ "-*&’’’’’’’’’’’’’’’’’’,0„
2__inference_time_distributed_2_layer_call_fn_34383oZ[C¢@
9¢6
,)
inputs’’’’’’’’’/3 
p

 
Ŗ "$!’’’’’’’’’,0„
2__inference_time_distributed_2_layer_call_fn_34390oZ[C¢@
9¢6
,)
inputs’’’’’’’’’/3 
p 

 
Ŗ "$!’’’’’’’’’,0Ü
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_34408L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’,0
p

 
Ŗ ":¢7
0-
0&’’’’’’’’’’’’’’’’’’	
 Ü
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_34426L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’,0
p 

 
Ŗ ":¢7
0-
0&’’’’’’’’’’’’’’’’’’	
 É
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_34445xC¢@
9¢6
,)
inputs’’’’’’’’’,0
p

 
Ŗ "1¢.
'$
0’’’’’’’’’	
 É
M__inference_time_distributed_3_layer_call_and_return_conditional_losses_34454xC¢@
9¢6
,)
inputs’’’’’’’’’,0
p 

 
Ŗ "1¢.
'$
0’’’’’’’’’	
 ³
2__inference_time_distributed_3_layer_call_fn_34431}L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’,0
p

 
Ŗ "-*&’’’’’’’’’’’’’’’’’’	³
2__inference_time_distributed_3_layer_call_fn_34436}L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’,0
p 

 
Ŗ "-*&’’’’’’’’’’’’’’’’’’	”
2__inference_time_distributed_3_layer_call_fn_34459kC¢@
9¢6
,)
inputs’’’’’’’’’,0
p

 
Ŗ "$!’’’’’’’’’	”
2__inference_time_distributed_3_layer_call_fn_34464kC¢@
9¢6
,)
inputs’’’’’’’’’,0
p 

 
Ŗ "$!’’’’’’’’’	É
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_34488xC¢@
9¢6
,)
inputs’’’’’’’’’	
p

 
Ŗ "1¢.
'$
0’’’’’’’’’	
 É
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_34497xC¢@
9¢6
,)
inputs’’’’’’’’’	
p 

 
Ŗ "1¢.
'$
0’’’’’’’’’	
 Ü
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_34540L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’	
p

 
Ŗ ":¢7
0-
0&’’’’’’’’’’’’’’’’’’	
 Ü
M__inference_time_distributed_4_layer_call_and_return_conditional_losses_34558L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’	
p 

 
Ŗ ":¢7
0-
0&’’’’’’’’’’’’’’’’’’	
 ”
2__inference_time_distributed_4_layer_call_fn_34502kC¢@
9¢6
,)
inputs’’’’’’’’’	
p

 
Ŗ "$!’’’’’’’’’	”
2__inference_time_distributed_4_layer_call_fn_34507kC¢@
9¢6
,)
inputs’’’’’’’’’	
p 

 
Ŗ "$!’’’’’’’’’	³
2__inference_time_distributed_4_layer_call_fn_34563}L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’	
p

 
Ŗ "-*&’’’’’’’’’’’’’’’’’’	³
2__inference_time_distributed_4_layer_call_fn_34568}L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’	
p 

 
Ŗ "-*&’’’’’’’’’’’’’’’’’’	Õ
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_34585L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’	
p

 
Ŗ "3¢0
)&
0’’’’’’’’’’’’’’’’’’	
 Õ
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_34602L¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’	
p 

 
Ŗ "3¢0
)&
0’’’’’’’’’’’’’’’’’’	
 Ā
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_34622qC¢@
9¢6
,)
inputs’’’’’’’’’	
p

 
Ŗ "*¢'
 
0’’’’’’’’’	
 Ā
M__inference_time_distributed_5_layer_call_and_return_conditional_losses_34632qC¢@
9¢6
,)
inputs’’’’’’’’’	
p 

 
Ŗ "*¢'
 
0’’’’’’’’’	
 ¬
2__inference_time_distributed_5_layer_call_fn_34607vL¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’	
p

 
Ŗ "&#’’’’’’’’’’’’’’’’’’	¬
2__inference_time_distributed_5_layer_call_fn_34612vL¢I
B¢?
52
inputs&’’’’’’’’’’’’’’’’’’	
p 

 
Ŗ "&#’’’’’’’’’’’’’’’’’’	
2__inference_time_distributed_5_layer_call_fn_34637dC¢@
9¢6
,)
inputs’’’’’’’’’	
p

 
Ŗ "’’’’’’’’’	
2__inference_time_distributed_5_layer_call_fn_34642dC¢@
9¢6
,)
inputs’’’’’’’’’	
p 

 
Ŗ "’’’’’’’’’	ā
K__inference_time_distributed_layer_call_and_return_conditional_losses_34128XYN¢K
D¢A
74
inputs(’’’’’’’’’’’’’’’’’’ī
p

 
Ŗ "<¢9
2/
0(’’’’’’’’’’’’’’’’’’ė 
 ā
K__inference_time_distributed_layer_call_and_return_conditional_losses_34152XYN¢K
D¢A
74
inputs(’’’’’’’’’’’’’’’’’’ī
p 

 
Ŗ "<¢9
2/
0(’’’’’’’’’’’’’’’’’’ė 
 Š
K__inference_time_distributed_layer_call_and_return_conditional_losses_34181XYE¢B
;¢8
.+
inputs’’’’’’’’’ī
p

 
Ŗ "3¢0
)&
0’’’’’’’’’ė 
 Š
K__inference_time_distributed_layer_call_and_return_conditional_losses_34196XYE¢B
;¢8
.+
inputs’’’’’’’’’ī
p 

 
Ŗ "3¢0
)&
0’’’’’’’’’ė 
 ŗ
0__inference_time_distributed_layer_call_fn_34159XYN¢K
D¢A
74
inputs(’’’’’’’’’’’’’’’’’’ī
p

 
Ŗ "/,(’’’’’’’’’’’’’’’’’’ė ŗ
0__inference_time_distributed_layer_call_fn_34166XYN¢K
D¢A
74
inputs(’’’’’’’’’’’’’’’’’’ī
p 

 
Ŗ "/,(’’’’’’’’’’’’’’’’’’ė §
0__inference_time_distributed_layer_call_fn_34203sXYE¢B
;¢8
.+
inputs’’’’’’’’’ī
p

 
Ŗ "&#’’’’’’’’’ė §
0__inference_time_distributed_layer_call_fn_34210sXYE¢B
;¢8
.+
inputs’’’’’’’’’ī
p 

 
Ŗ "&#’’’’’’’’’ė 